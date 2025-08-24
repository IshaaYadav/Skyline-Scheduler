# cli.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from load import load_flights

def main():
    if len(sys.argv) < 2:
        print("Usage: python cli.py <Flight_Data.xlsx>")
        sys.exit(1)

    filepath = sys.argv[1]
    df = load_flights(filepath)
    out_path = "outputs/flights_clean.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Cleaned flights saved to {out_path}")


if __name__ == "__main__":
    main()
