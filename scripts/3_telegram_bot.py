import google.generativeai as genai
import pandas as pd
from scipy import sparse
import numpy as np
from collections import defaultdict, deque
import asyncio
import pickle
import unicodedata
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from dotenv import load_dotenv
import os

load_dotenv()
GENAI_API_KEY   = os.getenv("GENAI_API_KEY", "")
TELEGRAM_TOKEN  = os.getenv("TELEGRAM_TOKEN", "")
BOT_SECRET      = os.getenv("BOT_SECRET", "")

# GEMINI CHARACTERS

PERSONA_EVERY_N = 4 # 0 means deactivated

PERSONAJES = ["Gemini"]


def _build_prompt_personaje(personaje, contexto_msgs):
    contexto = "\n".join(f"- {m}" for m in contexto_msgs[-5:])
    return (
        f"Eres {personaje}, un miembro del chat grupal. "
        f"Responde con un único mensaje coherente al contexto y tono reciente del chat.\n"
        f"No expliques tu razonamiento, no pongas prefijos de rol, no te disculpes.\n"
        f"Contexto:\n{contexto}\n"
        f"Respuesta:"
    )

def personaje_responde_sync(contexto_msgs, personaje):
    genai.configure(api_key=GENAI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash") # alternatives: "gemini-1.5-flash-8b", "gemini-1.5-pro"
    import random
    if not personaje:
        personaje = random.choice(PERSONAJES)
    prompt = _build_prompt_personaje(personaje, contexto_msgs)
    resp = model.generate_content(prompt)
    text = (resp.text or "").strip()
    return personaje, text

# PATHS
BASE = "C:/Users/Iriondo Delgado/Documents/chatbot"
INDEX_DIR = f"{BASE}/index"

VECT_WORDS_PKL   = f"{INDEX_DIR}/vectorizer_words.pkl"
VECT_CHARS_PKL   = f"{INDEX_DIR}/vectorizer_chars.pkl"
TFIDF_WORDS_NPZ  = f"{INDEX_DIR}/tfidf_words.npz"
TFIDF_CHARS_NPZ  = f"{INDEX_DIR}/tfidf_chars.npz"
PAIRS_PARQUET    = f"{INDEX_DIR}/pairs.parquet"
XW_PAIRS_NPZ     = f"{INDEX_DIR}/Xw_pairs.npz"
XC_PAIRS_NPZ     = f"{INDEX_DIR}/Xc_pairs.npz"

# PARAMETERS
TAIL_K = 5
TARGET_RESPONDER = "auto"
ALPHA_WORDS = 0.7
ALPHA_CHARS = 0.3
LAMBDA = 0.8
TOP_K_PAIRS = 800
SAMPLE_TOP_N = 8
RANDOM_SEED = None
MIN_REPLY_LEN = 8
MAX_REPLY_LEN = 200
CONTEXT_DECAY = 0.75
REMOVE_ACCENTS_FOR_DEDUP = True
COOLDOWN_LAST_N_REPLIES = 40
MIN_SCORE_Z = -0.1
STOP = {"de","la","el","y","o","a","en","que","un","una","es","si","no","al","del","lo",
        "los","las","por","con","para","se","me","te","le","mi","tu","su","ya","pero","mas",
        "muy","como","porque","qué","qué","ese","esa","esto","esta","eso","hay","soy","eres",
        "estoy","estás","está","estamos","estáis","están"}
BURST_MESSAGES = 2


def load_cache():
    with open(VECT_WORDS_PKL, "rb") as f:
        vw = pickle.load(f)
    with open(VECT_CHARS_PKL, "rb") as f:
        vc = pickle.load(f)
    Xw_msgs = sparse.load_npz(TFIDF_WORDS_NPZ)
    Xc_msgs = sparse.load_npz(TFIDF_CHARS_NPZ)
    pairs = pd.read_parquet(PAIRS_PARQUET)
    Xw_pairs = sparse.load_npz(XW_PAIRS_NPZ)
    Xc_pairs = sparse.load_npz(XC_PAIRS_NPZ)
    return vw, vc, Xw_msgs, Xc_msgs, pairs, Xw_pairs, Xc_pairs

VW, VC, XW_MSGS, XC_MSGS, PAIRS, XW_PAIRS, XC_PAIRS = load_cache()

# MOTOR
def normalize_for_dedup(text):
    t = (text or "").strip().lower()
    t = " ".join(t.split())
    t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    return t

def keywords(s: str) -> set:
    t = unicodedata.normalize("NFD", (s or "").lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    toks = [tok.strip(".,!?¡¿:;()[]\"'") for tok in t.split()]
    return {tok for tok in toks if tok and tok not in STOP}

def cosine_scores(X, q_vec):
    sim = X.dot(q_vec.T)
    return np.asarray(sim.todense()).ravel()

def vectorize_context(messages, vw, vc, decay=CONTEXT_DECAY):
    msgs = [m for m in messages if isinstance(m, str) and m.strip()]
    if not msgs:
        return [], [], []
    n = len(msgs)
    weights = [decay**(n-1-i) for i in range(n)]
    s = sum(weights)
    weights = [w/s for w in weights] if s > 0 else [1.0/n]*n
    qw_list = [vw.transform([m]) for m in msgs]
    qc_list = [vc.transform([m]) for m in msgs]
    return qw_list, qc_list, weights

def score_accumulate(Xw, Xc, qw_list, qc_list, weights):
    n_docs = Xw.shape[0]
    score = np.zeros(n_docs, dtype=np.float32)
    for qw, qc, w in zip(qw_list, qc_list, weights):
        sw = cosine_scores(Xw, qw)
        sc = cosine_scores(Xc, qc)
        score += w * (ALPHA_WORDS * sw + ALPHA_CHARS * sc)
    return score

def zscore(x):
    mu, sigma = x.mean(), x.std()
    return (x - mu) / (sigma + 1e-8)

def sample_one(top_df, top_n=5, seed=None):
    top = top_df.head(top_n).copy()
    if top.empty:
        return None
    w = top["score"].clip(lower=0).to_numpy()
    p = (w / w.sum()) if w.sum() > 0 else None
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(top), p=p)
    return top.iloc[idx]

HISTORY  = defaultdict(list)
COOLDOWN = defaultdict(lambda: deque(maxlen=COOLDOWN_LAST_N_REPLIES))
TURNS    = defaultdict(int)

def generate_replies(chat_id: int, user_msg: str) -> list[str]:
    HISTORY[chat_id].append({"speaker": "Tú", "text": user_msg})

    tail = HISTORY[chat_id][-TAIL_K:] if len(HISTORY[chat_id]) >= TAIL_K else HISTORY[chat_id][:]
    context_msgs = [m["text"] for m in tail]
    last_speaker = tail[-1]["speaker"] if tail else None

    # vectorize context
    qw_list, qc_list, weights = vectorize_context(context_msgs, VW, VC, CONTEXT_DECAY)
    if not qw_list:
        return ["(Escribe algo para iniciar.)"]

    # scores
    score_pairs = score_accumulate(XW_PAIRS, XC_PAIRS, qw_list, qc_list, weights)
    score_msgs  = score_accumulate(XW_MSGS,  XC_MSGS,  qw_list, qc_list, weights)

    cands = PAIRS.copy()
    cands["score_pairs"] = score_pairs

    reply_scores = score_msgs[cands["reply_id"].values]
    cands["score_hybrid"] = LAMBDA * zscore(cands["score_pairs"]) + (1.0 - LAMBDA) * zscore(reply_scores)

    cands = cands.sort_values("score_hybrid", ascending=False).head(TOP_K_PAIRS).reset_index(drop=True)
    cands = cands[cands["score_hybrid"] >= MIN_SCORE_Z]

    # avoid repeating the context
    ctx_set = {normalize_for_dedup(m) for m in context_msgs if isinstance(m, str)}
    cands["__norm_reply"] = cands["reply_text"].apply(normalize_for_dedup)
    cands = cands[~cands["__norm_reply"].isin(ctx_set)].drop(columns="__norm_reply", errors="ignore")

    # theme
    ctx_kw = set().union(*[keywords(m) for m in context_msgs if isinstance(m, str)]) if context_msgs else set()
    if ctx_kw:
        cands = cands[cands["reply_text"].apply(lambda txt: len(keywords(txt) & ctx_kw) > 0)]

    cands = cands.drop_duplicates(subset="reply_text", keep="first")

    # cooldown
    if len(COOLDOWN[chat_id]) > 0:
        cands = cands[~cands["reply_id"].isin(COOLDOWN[chat_id])]

    if cands.empty:
        return ["(No encuentro una respuesta adecuada; prueba a reformular o cambia de tema.)"]

    # 
    burst_pool = cands.rename(columns={"score_hybrid": "score"}).copy()
    picked_rows = []
    for _ in range(BURST_MESSAGES):
        if burst_pool.empty:
            break
        s = sample_one(burst_pool, top_n=SAMPLE_TOP_N, seed=RANDOM_SEED)
        if s is None:
            break
        picked_rows.append(s)
        rid = int(s["reply_id"])
        burst_pool = burst_pool[burst_pool["reply_id"] != rid]

    if not picked_rows:
        return ["(No hay candidatos válidos tras filtros.)"]

    # history and cooldown register
    replies = []
    for s in picked_rows:
        reply_text = (s["reply_text"] or "").strip()
        reply_speaker = s["reply_speaker"]
        reply_id = int(s["reply_id"])
        replies.append(f"{reply_speaker}: {reply_text}")
        HISTORY[chat_id].append({"speaker": reply_speaker, "text": reply_text})
        COOLDOWN[chat_id].append(reply_id)

    return replies


# TELEGRAM

AUTH = set()

async def login(update, context):
    chat_id = update.effective_chat.id
    args = context.args or []
    if len(args) != 1:
        return await update.message.reply_text("Escribe /login <secreto>")
    if (args[0].strip() == BOT_SECRET):
        AUTH.add(chat_id)
        return await update.message.reply_text("Acceso concedido. ¡Ya puedes chatear!")
    return await update.message.reply_text("Secreto incorrecto.")

async def logout(update, context):
    chat_id = update.effective_chat.id
    AUTH.discard(chat_id)
    await update.message.reply_text("Sesión cerrada.")

async def start(update, context):
    await update.message.reply_text(
        "¡Hola! Soy el imitador del grupo.\n"
        "Comandos: /help, /reset, /pj [Nombre]\n"
        "Consejo: usa /pj para meter un personaje inventado."
    )

async def help_cmd(update, context):
    await update.message.reply_text(
        "Uso:\n"
        "- Escríbeme y responderé con 1–2 réplicas reales del chat.\n"
        "- /pj [Nombre] hace que responda un personaje inventado (si está activado).\n"
        "- Mantengo memoria por chat con los últimos mensajes (no persistente).\n"
    )

async def reset(update, context):
    chat_id = update.effective_chat.id
    HISTORY.pop(chat_id, None)
    COOLDOWN.pop(chat_id, None)
    TURNS.pop(chat_id, None)
    await update.message.reply_text("He borrado el historial para este chat. ¡Empezamos de cero!")

async def cmd_personaje(update, context):
    chat_id = update.effective_chat.id

    nombre = " ".join(context.args).strip() if context.args else None

    tail = [m for m in HISTORY[chat_id] if m["speaker"] != "Tú"][-TAIL_K:]
    context_msgs = [m["text"] for m in tail] or ([HISTORY[chat_id][-1]["text"]] if HISTORY[chat_id] else [])

    try:
        pj, txt = await asyncio.to_thread(personaje_responde_sync, context_msgs, nombre if nombre else None)
        HISTORY[chat_id].append({"speaker": pj, "text": txt})
        await update.message.reply_text(f"{pj}: {txt}")
    except Exception as e:
        await update.message.reply_text(f"(LLM personaje no disponible: {e})")

async def on_message(update, context):
    chat_id = update.effective_chat.id
    if chat_id not in AUTH:
        return await update.message.reply_text("Necesitas /login <secreto> para usar el bot.")
    text = (update.message.text or "").strip()
    if not text:
        return

    TURNS[chat_id] += 1

    if PERSONA_EVERY_N and TURNS[chat_id] % PERSONA_EVERY_N == 0:
        HISTORY[chat_id].append({"speaker": "Tú", "text": text})
        tail = [m for m in HISTORY[chat_id] if m["speaker"] != "Tú"][-TAIL_K:]
        context_msgs = [m["text"] for m in tail] or [text]
        try:
            pj, txt = await asyncio.to_thread(personaje_responde_sync, context_msgs, None)
            HISTORY[chat_id].append({"speaker": pj, "text": txt})
            await update.message.reply_text(f"{pj}: {txt}")
            return
        except Exception as e:
            await update.message.reply_text(f"(LLM personaje no disponible: {e}. Uso motor clásico.)")

    replies = generate_replies(chat_id, text)
    for r in replies:
        await update.message.reply_text(r)

def main():
    token = TELEGRAM_TOKEN

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("login", login))
    app.add_handler(CommandHandler("logout", logout))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("pj", cmd_personaje))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    print("Bot arrancado. Requiere login. Ctrl+C para parar.")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()