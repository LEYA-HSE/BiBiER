import sys
import os
from generate_synthetic_dataset import generate_from_emotion_csv

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Использование: python run_generation.py path/to/file.csv [num_processes] [device]")
        sys.exit(1)

    csv_path = sys.argv[1]
    num_processes = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("NUM_DIA_PROCESSES", 1))
    device = sys.argv[3] if len(sys.argv) > 3 else "cuda"

    filename = os.path.basename(csv_path)
    try:
        emotion = filename.split("_")[2]
    except IndexError:
        emotion = "unknown"

    print(f"🧪 CSV: {csv_path}")
    print(f"💻 Устройство: {device}")
    print(f"🔧 Процессов: {num_processes}")
    print(f"🎭 Эмоция: {emotion}")

    generate_from_emotion_csv(
        csv_path=csv_path,
        emotion=emotion,
        output_dir="tts_synthetic_final",
        device=device,
        max_samples=None,
        num_processes=num_processes
    )
