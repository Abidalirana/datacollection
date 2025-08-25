PROJECT 04:

fundedflow/                  ← Your main project folder
└── my_ai_project/           ← All project code lives here
    │── main.py              # Entry point / orchestrator
    │── requirements.txt     # Python dependencies
    │── config.py            # API keys, DB settings
    │── README.md
    │── .env
    │── .gitignore
    │
    ├── workflow/
    │   ├── __init__.py
    │   ├── pipeline.py      # Collector → DB → embeddings → LLM → agents
    │   └── utils.py         # Logging, helpers
    │
    ├── data_collector/
    │   ├── __init__.py
    │   ├── orchestrator_collector.py
    │   ├── user_profile.py
    │   ├── emotion_tracker.py
    │   ├── trade_data.py
    │   ├── engagement_logger.py
    │   └── ai_interaction_logger.py
    │
    ├── database/
    │   └── models.py     # DB connection + tables
    │   |--- __init__.py   
        |--create_db.py  
    ├── llm/
    │   ├── __init__.py
    │   ├── llm_client.py
    │   ├── embedding_client.py
    │   ├── prompt_templates.py
    │   └── orchestrator_llm.py
    │
    └── myagents/
        ├── __init__.py
        ├── mindset_coach_agent.py
        ├── trade_therapist_agent.py
        ├── recovery_planner_agent.py
        ├── risk_manager_agent.py
        ├── onboarding_coach_agent.py
        └── propfirm_intelligence_agent.py
        |-----orchestrator.py





====================================================================================


