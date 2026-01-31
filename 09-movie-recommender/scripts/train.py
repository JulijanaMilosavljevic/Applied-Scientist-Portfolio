from src.data_loader import load_ratings
from src.preprocessing import train_test_split_by_user, create_user_item_matrix


def main():
    ratings = load_ratings()

    train_df, test_df = train_test_split_by_user(ratings, test_ratio=0.2)

    print("✅ Full ratings:", ratings.shape)
    print("✅ Train:", train_df.shape)
    print("✅ Test:", test_df.shape)

    # sanity: no overlap for same user-item pairs ideally (can overlap if user rated same item twice; rare)
    train_pairs = set(zip(train_df["user_id"], train_df["item_id"]))
    test_pairs = set(zip(test_df["user_id"], test_df["item_id"]))
    overlap = len(train_pairs & test_pairs)
    print("Overlap user-item pairs:", overlap)

    user_item_train = create_user_item_matrix(train_df)
    print("✅ User-item train matrix shape:", user_item_train.shape)

    # sparsity
    nnz = (user_item_train.values != 0).sum()
    total = user_item_train.shape[0] * user_item_train.shape[1]
    sparsity = 1 - nnz / total
    print(f"✅ Sparsity (train matrix): {sparsity:.2%}")


if __name__ == "__main__":
    main()
