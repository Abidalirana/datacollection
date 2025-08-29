# run_pipeline.py

import subprocess
import sys
import asyncio

# Step runner helper
def run_step(description, command):
    print(f"\n🚀 Starting: {description}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"❌ Error in {description}, stopping pipeline.")
        sys.exit(1)
    print(f"✅ Finished: {description}")

# Fully async wrapper to call agent after retrieval
async def run_agent_with_query(user_query: str):
    from ai_project.llm_agent.my_agent import run_my_agent
    response = await run_my_agent(user_query)
    return response

def main(user_query: str):
    # 1️⃣ Data Collection
    run_step("Data Collection", "python -m ai_project.data_collector.orchestrator_collector")

    # 2️⃣ Preprocessing & Feature Engineering
    run_step("Preprocessing", "python -m ai_project.data_processing.run_preprocess")

    # 3️⃣ ML Orchestration
    run_step("Machine Learning Orchestration", "python -m ai_project.ml_models.ml_orchestrator")

    # 4️⃣ Embedding Ingestion
    run_step("Embedding Ingestion", "python -m ai_project.embedding.embedding_ingest")

    # 5️⃣ Embedding Retrieval
    run_step("Embedding Retrieval", f"python -m ai_project.embedding.embedding_retrieval \"{user_query}\"")

    # 6️⃣ Run LLM agent with user query + retrieved context
    print("\n🤖 Running LLM agent with your query...\n")
    response = asyncio.run(run_agent_with_query(user_query))
    print(f"💬 LLM Agent Response:\n{response}")

    print("\n🎉 Full pipeline completed successfully!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️ Please provide a query. Example: python run_pipeline.py 'show my last 10 trades'")
        sys.exit(1)

    user_query = sys.argv[1]
    main(user_query)
