

fundedflow/
└── my_ai_project/
    │── main.py                 # Entry point: orchestrator for data collection + agents
    │── requirements.txt        # Python dependencies
    │── config.py               # DB and API keys
    │── README.md
    │── .env
    │── .gitignore
    │
    ├── database/
    │   ├── __init__.py
    │   ├── models.py           # DB tables: users, trades, emotions, journals, etc.
    │   └── create_db.py        # Initialize DB
    │
    ├── data_collector/
    │   ├── __init__.py
    │   ├── orchestrator_collector.py  # Orchestrator for all data collection modules
    │   ├── user_profile.py
    │   ├── emotion_tracker.py
    │   ├── trade_data.py
    │   ├── engagement_logger.py
    │   ├── journal_logger.py
    │   └── ai_interaction_logger.py
    │
    ├── llm/
    │   ├── __init__.py
    │   ├── llm_client.py             # OpenAI / GPT client
    │   ├── embedding_client.py       # Pinecone / Weaviate
    │   ├── prompt_templates.py
    │   └── orchestrator_llm.py      # Orchestrator for embeddings + prompts
    │
    └── agents/
        ├── __init__.py
        ├── mindset_coach_agent.py
        ├── trade_therapist_agent.py
        ├── recovery_planner_agent.py
        ├── risk_manager_agent.py
        ├── onboarding_coach_agent.py
        ├── propfirm_intelligence_agent.py
        └── orchestrator.py          # Orchestrator for all agents
==============================================================================================================================================================================================================================














==
==========================================================

01--for running the collctor files
python -m data_collector.orchestrator_collector
or
python -m ai_project.data_collector.orchestrator_collector
===================
for creation tables at a rute folder...
python -m ai_project.database.create_db
============

