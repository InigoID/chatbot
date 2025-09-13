from datetime import datetime
import json, re

# Message grammar

PATTERN = re.compile(r'^(\d{1,2}/\d{1,2}/\d{2,4}),\s+(\d{1,2}:\d{2})\s+-\s+(.*?)(?::\s*(.*))?$')

# Text to remove

SYSTEM_HINTS = [
    "Los mensajes y llamadas están cifrados de extremo a extremo",
    "cifrado de extremo a extremo",
    "Cambiaste el asunto",
    "cambió el asunto",
    "cambió la descripción del grupo",
    "cambió la foto del grupo",
    "cambió el nombre del grupo",
    "creaste este grupo",
    "creó el grupo",
    "añadió",
    "añadiste",
    "fue añadido",
    "salió del grupo",
    "se unió",
    "invitación del grupo",
    "Solo los administradores pueden",
    "Este grupo ahora permite",
    "Mensajes temporales",
    "El administrador cambió",
]

TEXT_PLACEHOLDERS = [
    "<multimedia omitido>",
    "sticker omitido",
    "mensaje eliminado",
    "este mensaje fue eliminado",
    "mensajes temporales activados",
    "mensajes temporales desactivados",
]

def looks_like_system(name_or_text):
    t = (name_or_text or "").strip().lower()
    if not t:
        return True
    return any(h.lower() in t for h in SYSTEM_HINTS)

def text_is_placeholder(txt):
    t = (txt or "").strip().lower()
    if not t:
        return True
    return any(ph.lower() == t for ph in TEXT_PLACEHOLDERS)

# Transform whatsapp chat text file to jsonl

def main(in_path, out_path):
    messages, current, messages_number = [], None, 0
    with open(in_path, "r", encoding="utf-8-sig", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\r\n")
            m = PATTERN.match(line)

            if m:
                # End previous message
                if current:
                    messages.append(current)
                    current = None
                    messages_number += 1

                d, t, who, text = m.groups()
                dt = datetime.strptime(f"{d} {t}", "%d/%m/%y %H:%M")

                # Go to the next message
                if text is None:
                    continue
                if looks_like_system(who) or text_is_placeholder(text):
                    continue
                current = {"timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"), "speaker": who.strip(), "text": text.strip()}

            elif current:
                current["text"] += ("\n" if current["text"] else "") + line
        
    if current:
        messages.append(current)

    with open(out_path, "w", encoding="utf-8") as out:
        for m in messages:
            out.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"JSONL of {messages_number} messages generated")

# Replace with correct paths

if __name__ == "__main__":
    main("C:/Users/Iriondo Delgado/Documents/chatbot/raw/Chat de Whatsapp con desgana.txt", "C:/Users/Iriondo Delgado/Documents/chatbot/processed/messages.jsonl")
