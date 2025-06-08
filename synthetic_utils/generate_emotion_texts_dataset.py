import json
import random
import pandas as pd
import re
from datetime import timedelta
from pathlib import Path

# === Загрузка шаблонов ===
def load_templates_json(templates_dir, emotion):
    path = Path(templates_dir) / f"{emotion}.json"
    if not path.exists():
        raise FileNotFoundError(f"Шаблон для эмоции '{emotion}' не найден: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# === Генерация текстов с учётом seed и антидубликатов ===
def generate_emotion_batch(n, template_data, seed=None):
    if seed is not None:
        random.seed(seed)

    subjects       = template_data["subjects"]
    verbs          = template_data["verbs"]
    contexts       = template_data["contexts"]
    interjections  = template_data.get("interjections", [""])
    templates      = template_data["templates"]

    # Допустимые звуковые метки DIA‑TTS
    dia_tags = {
        "(laughs)", "(clears throat)", "(sighs)", "(gasps)", "(coughs)",
        "(singing)", "(sings)", "(mumbles)", "(beep)", "(groans)", "(sniffs)",
        "(claps)", "(screams)", "(inhales)", "(exhales)", "(applause)",
        "(burps)", "(humming)", "(sneezes)", "(chuckle)", "(whistles)"
    }

    def has_tag(text):      return any(tag in text for tag in dia_tags)
    def remove_tags(text):
        for tag in dia_tags:
            text = text.replace(tag, "")
        return text.strip()

    phrases, attempts = set(), 0
    max_attempts = n * 50

    while len(phrases) < n and attempts < max_attempts:
        s, v = random.choice(subjects), random.choice(verbs)
        c, i = random.choice(contexts), random.choice(interjections)
        t     = random.choice(templates)

        # ▸ Разрешаем максимум одну звуковую метку на фразу
        if has_tag(i) and has_tag(c):
            if random.random() < .5:
                c = remove_tags(c)
            else:
                i = remove_tags(i)

        phrase = t.format(s=s, v=v, c=c, i=i)

        # --- Очистка без разрушения многоточий ---------------------------
        # 1) убрать пробелы перед знаками пунктуации
        phrase = re.sub(r"\s+([,.!?])", r"\1", phrase)
        # 2) превратить двойную точку, КОТОРАЯ не часть троеточия, в одну
        phrase = re.sub(r"(?<!\.)\.\.(?!\.)", ".", phrase)
        # 3) вставить пробел, если после метки сразу идёт слово
        phrase = re.sub(r"\)(?=\w)", ") ", phrase)
        # 4) схлопнуть множественные пробелы и обрезать края
        phrase = re.sub(r"\s{2,}", " ", phrase).strip()
        # ------------------------------------------------------------------

        if phrase not in phrases:
            phrases.add(phrase)
        attempts += 1

    if len(phrases) < n:
        print(f"⚠️ Только {len(phrases)} уникальных фраз из {n} запрошенных — возможно, исчерпан пул шаблонов.")

    return list(phrases)

# === Генерация временных меток ===
def generate_dummy_timestamps(n):
    base_time, result = timedelta(), []
    for idx in range(n):
        start = base_time + timedelta(seconds=idx * 6)
        end   = start + timedelta(seconds=5)
        result.append((
            str(start).split(".")[0] + ",000",
            str(end).split(".")[0]   + ",000"
        ))
    return result

# === Финальная сборка и сохранение CSV ===
def create_emotion_csv(template_path, emotion_label, out_file, n=1000, seed=None):
    data     = load_templates_json(template_path, emotion_label)
    phrases  = generate_emotion_batch(n, data, seed)
    timeline = generate_dummy_timestamps(n)

    emotions   = ["neutral", "happy", "sad", "anger", "surprise", "disgust", "fear"]
    label_mask = {e: float(e == emotion_label) for e in emotions}

    df = pd.DataFrame({
        "video_name": [f"dia_{emotion_label}_utt{i}_synt" for i in range(n)],
        "start_time": [s for s, _ in timeline],
        "end_time"  : [e for _, e in timeline],
        "sentiment" : [0] * n,
        **{e: [label_mask[e]] * n for e in emotions},
        "text"      : phrases
    })

    df.to_csv(out_file, index=False)
    print(f"✅ Сохранено {len(df)} строк → {out_file}")

    # --- Проверка дубликатов ---
    dupes = df[df.duplicated("text", keep=False)]
    if not dupes.empty:
        dupe_file = Path(out_file).with_name(f"duplicates_{emotion_label}.csv")
        dupes.to_csv(dupe_file, index=False)
        print(f"⚠️ Найдено {len(dupes)} повторов → {dupe_file}")
    else:
        print("✅ Дубликатов нет.")

# === Точка входа ===
if __name__ == "__main__":
    emotion_config = {
        "anger":    3600,
        "disgust":  4438,
        "fear":     4441,
        "happy":    2966,
        "sad":      4026,
        "surprise": 3504
    }

    seed, template_path, out_dir = 42, "emotion_templates", "synthetic_data"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for emotion, n in emotion_config.items():
        out_csv = Path(out_dir) / f"meld_synthetic_{emotion}_{n}.csv"
        print(f"\n🔄 Генерация: {emotion} ({n} фраз)")
        create_emotion_csv(template_path, emotion, str(out_csv), n, seed)
