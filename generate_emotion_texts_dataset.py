import json
import random
import pandas as pd
from datetime import timedelta
from pathlib import Path

# === Загрузка шаблонов из templates/<emotion>.json ===
def load_templates_json(templates_dir, emotion):
    path = Path(templates_dir) / f"{emotion}.json"
    if not path.exists():
        raise FileNotFoundError(f"Шаблон для эмоции '{emotion}' не найден: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# === Генерация уникальных фраз по шаблонам с антишаблонным стилем ===
def generate_emotion_batch(n, template_data):
    subjects = template_data["subjects"]
    verbs = template_data["verbs"]
    contexts = template_data["contexts"]
    interjections = template_data.get("interjections", [""])
    templates = template_data["templates"]

    phrases = set()
    attempts = 0
    max_attempts = n * 50

    while len(phrases) < n and attempts < max_attempts:
        s = random.choice(subjects)
        v = random.choice(verbs)
        c = random.choice(contexts)
        i = random.choice(interjections)
        t = random.choice(templates)
        phrase = t.format(s=s, v=v, c=c, i=i).replace("..", ".").replace(" ,", ",").strip()
        if phrase not in phrases:
            phrases.add(phrase)
        attempts += 1

    if len(phrases) < n:
        print(f"⚠️ Только {len(phrases)} уникальных фраз из {n} запрошенных — возможно, исчерпан пул шаблонов.")

    return list(phrases)

# === Генерация фиктивных временных меток в формате MELD ===
def generate_dummy_timestamps(n):
    base_time = timedelta()
    start_end = []
    for i in range(n):
        start = base_time + timedelta(seconds=i * 6)
        end = start + timedelta(seconds=5)
        start_end.append((
            str(start).split(".")[0] + ",000",
            str(end).split(".")[0] + ",000"
        ))
    return start_end

# === Генерация CSV-датафрейма в стиле MELD ===
def create_emotion_csv(template_path, emotion_label, out_file, n=1000):
    data = load_templates_json(template_path, emotion_label)
    phrases = generate_emotion_batch(n, data)
    timestamps = generate_dummy_timestamps(n)

    emotions = ["neutral", "happy", "sad", "anger", "surprise", "disgust", "fear"]
    label_row = {e: 0.0 for e in emotions}
    label_row[emotion_label] = 1.0

    df = pd.DataFrame({
        "video_name": [f"dia_{emotion_label}_utt{i}_synt" for i in range(n)],
        "start_time": [s for s, _ in timestamps],
        "end_time": [e for _, e in timestamps],
        "sentiment": [0] * n,
        **{e: [label_row[e]] * n for e in emotions},
        "text": phrases
    })

    df.to_csv(out_file, index=False)
    print(f"✅ Сохранено {len(df)} строк в файл: {out_file}")

    # === Проверка на дубликаты ===
    dupes = df[df.duplicated(subset=["text"], keep=False)]
    if not dupes.empty:
        dupe_file = Path(out_file).with_name(f"duplicates_{emotion_label}.csv")
        dupes.to_csv(dupe_file, index=False)
        print(f"⚠️ Найдено {len(dupes)} повторов. Сохранено в: {dupe_file}")
    else:
        print("✅ Дубликаты не найдены.")


if __name__ == "__main__":
    # === Настройки генерации ===
    emotion_label = "surprise"
    n = 3504
    template_path = "emotion_templates"
    out_dir = "synthetic_data"

    # === Формирование пути к файлу ===
    Path(out_dir).mkdir(parents=True, exist_ok=True)  # Создаём папку, если нет
    out_file = Path(out_dir) / f"meld_synthetic_{emotion_label}_{n}.csv"

    # === Запуск генерации ===
    create_emotion_csv(
        template_path=template_path,
        emotion_label=emotion_label,
        out_file=str(out_file),
        n=n
    )
