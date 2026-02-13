from load_data import load_ebay_data, load_amazon_products

def explore_datasets():
    """Explore column names and data types"""
    
    # eBay dataset
    print("="*60)
    print("eBay Dataset")
    print("="*60)
    ebay = load_ebay_data()
    print("\nColumn names:")
    for i, col in enumerate(ebay.columns, 1):
        print(f"{i:2d}. {col}")
    
    print("\nSample data:")
    print(ebay.head(2))
    
    print("\nData types:")
    print(ebay.dtypes)
    
    # Amazon dataset
    print("\n" + "="*60)
    print("Amazon Dataset")
    print("="*60)
    amazon = load_amazon_products()
    print("\nColumn names:")
    for i, col in enumerate(amazon.columns, 1):
        print(f"{i:2d}. {col}")
    
    print("\nSample data:")
    print(amazon.head(2))
    
    print("\nData types:")
    print(amazon.dtypes)

if __name__ == "__main__":
    explore_datasets()