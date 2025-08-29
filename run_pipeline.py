# run_pipeline.py

import subprocess
import sys

def run_step(description, command):
    print(f"\n🚀 Starting: {description}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"❌ Error in {description}, stopping pipeline.")
        sys.exit(1)
    print(f"✅ Finished: {description}")

def main(user_query: str):
    # Step 1: Data Collection
    run_step("Data Collection", "python -m ai_project.data_collector.orchestrator_collector")

    # Step 2: Preprocessing
    run_step("Preprocessing", "python -m ai_project.data_processing.run_preprocess")

    # Step 3: ML Orchestrator
    run_step("Machine Learning Orchestration", "python -m ai_project.ml_models.ml_orchestrator")

    # Step 4: Embedding Ingestion
    run_step("Embedding Ingestion", "python -m ai_project.embedding.embedding_ingest")

    # Step 5: Retrieval (optional: pass query)
    run_step("Embedding Retrieval", f"python -m ai_project.embedding.embedding_retrieval \"{user_query}\"")

    print("\n🎉 Full pipeline completed successfully!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️ Please provide a query. Example: python run_pipeline.py 'show my last 10 trades'")
        sys.exit(1)

    user_query = sys.argv[1]
    main(user_query)
