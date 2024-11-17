from data_loaders import StatsmodelsLoader, CSVLoader, OnlineCSVLoader

def test_data_loaders():
    # Test StatsmodelsLoader
    print("\nTesting StatsmodelsLoader")
    stats_loader = StatsmodelsLoader("longley")  # Use "longley" dataset from statsmodels
    stats_loader.load_data()
    try:
        stats_loader.add_constant()
        print("X (with constant):", stats_loader.X[:5])
        print("y:", stats_loader.y[:5])
    except ValueError as e:
        print(e)

    # Test CSVLoader (local file)
    print("\nTesting CSVLoader")
    csv_loader = CSVLoader("local_dataset.csv")  # Replace with a valid local file path
    try:
        csv_loader.load_data()
        csv_loader.add_constant()
        print("X (with constant):", csv_loader.X[:5])
        print("y:", csv_loader.y[:5])
    except FileNotFoundError:
        print(f"Error: File '{csv_loader.file_path}' not found. Please provide a valid file path.")
    except ValueError as e:
        print(f"Error: {e}")

    # Test OnlineCSVLoader
    print("\nTesting OnlineCSVLoader")
    online_loader = OnlineCSVLoader("https://people.sc.fsu.edu/~jburkardt/data/csv/airtravel.csv")
    try:
        online_loader.load_data()
        online_loader.add_constant()
        print("X (with constant):", online_loader.X[:5])
        print("y:", online_loader.y[:5])
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    test_data_loaders()