from src.data_loader import load_ratings
from src.preprocessing import train_test_split_by_user, create_user_item_matrix
from src.model_itemcf import compute_item_similarity
from src.evaluation import evaluate_model,get_relevant_items, precision_at_k, recall_at_k
from src.model_popularity import compute_popularity, recommend_popular
def main():
    ratings = load_ratings()
    train_df, test_df = train_test_split_by_user(ratings, test_ratio=0.2)

    user_item_train = create_user_item_matrix(train_df)

    print("Computing item-item similarity...")
    sim_matrix = compute_item_similarity(user_item_train)
    pop_df = compute_popularity(train_df)
    popularity_list = pop_df["item_id"].tolist()
    precision, recall = evaluate_model(
        user_item_train=user_item_train,
        sim_matrix=sim_matrix,
        test_df=test_df,
        k=10,
        min_rating=4,
        popularity_list=popularity_list,
        min_history=5
    )

    print(f"\n✅ Precision@10: {precision:.4f}")
    print(f"✅ Recall@10:    {recall:.4f}")
     # Popularity baseline
    pop_df = compute_popularity(train_df)

    precisions = []
    recalls = []
    users = test_df["user_id"].unique()

    for user_id in users:
        relevant = get_relevant_items(test_df, user_id, min_rating=4)
        if len(relevant) == 0:
            continue

        recommended = recommend_popular(user_id, user_item_train, pop_df, top_k=10)
        precisions.append(precision_at_k(recommended, relevant, 10))
        recalls.append(recall_at_k(recommended, relevant, 10))

    print(f"\n✅ Popularity Precision@10: {sum(precisions)/len(precisions):.4f}")
    print(f"✅ Popularity Recall@10:    {sum(recalls)/len(recalls):.4f}")
    print("\n--- ItemCF + Fallback (grid) ---")
    for k in [5, 10, 20]:
        p, r = evaluate_model(
            user_item_train=user_item_train,
            sim_matrix=sim_matrix,
            test_df=test_df,
            k=k,
            min_rating=4,
            popularity_list=popularity_list,
            min_history=5
        )
        print(f"ItemCF+Fallback  Precision@{k}: {p:.4f} | Recall@{k}: {r:.4f}")

if __name__ == "__main__":
    main()
