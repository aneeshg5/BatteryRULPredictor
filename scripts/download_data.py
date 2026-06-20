import logging

from battery_rul.data.download import download_dataset


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    for path in download_dataset():
        print(path)


if __name__ == "__main__":
    main()
