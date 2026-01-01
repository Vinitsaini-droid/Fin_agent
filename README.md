🧠 Fin-Agent: A Modular, Self-Verifying AI Reasoning System



This repository contains a production-grade, agentic AI system designed for financial reasoning, retrieval-augmented intelligence, and explainable decision-making.



It is not a chatbot toy.



It is a controlled cognitive pipeline built around:



explicit reasoning stages,



verifiable outputs,



memory and personalization,



and strict cost / hallucination control.



The system runs entirely locally (or on your infra) and requires no external orchestration frameworks.



🧩 What This System Actually Is



Fin-Agent is a single-agent architecture with internal cognition layers, inspired by how humans reason:



User → Planner → Thinker → Verifier → Explainer → User





Each stage has a clear responsibility and is independently testable.



The agent:



retrieves information only when necessary



verifies its own answers before returning them



maintains long-term user context



avoids hallucinations by construction, not by prompt hacks



🗂️ Repository Structure

Fin\_agent/

│

├── chatbot\_ui.py              # Streamlit interface (entry point)

│

├── agent/

│   ├── meta\_agent.py          # Routing + orchestration brain

│   ├── planner.py             # Task decomposition

│   ├── thinker.py             # Retrieval + reasoning

│   ├── verifier.py            # Factual \& compliance validation

│   ├── explainer.py           # Final answer synthesis

│   └── schemas.py             # Typed schemas for all components

│

├── memory/

│   ├── memory\_manager.py      # Persistent user memory

│   ├── chat\_summarizer.py     # Conversation compression

│   └── user\_profile\_store.py  # Preferences \& behavioral traits

│

├── retrieval/

│   ├── pinecone\_client.py     # Vector DB access

│   ├── semantic\_cache.py      # Query-level caching

│   ├── query\_refiner.py       # Query rewriting / HyDE

│   └── context\_compressor.py  # Token-efficient summarization

│

├── prompts/

│   ├── system\_base.txt

│   ├── planner\_prompt.txt

│   ├── thinker\_prompt.txt

│   ├── verifier\_prompt.txt

│   └── explainer\_prompt.txt

│

├── evaluation/

│   ├── ragas\_runner.py        # Automated evals

│   ├── aspect\_critics.py      # Domain \& logic critics

│   └── trace\_logger.py

│

├── config/

│   ├── settings.py

│   ├── compliance\_rules.py

│   └── token\_budgets.py

│

├── utils/

│   ├── llm\_client.py

│   ├── json\_utils.py

│   ├── similarity.py

│   └── logging.py

│

├── main\_agent.py              # Core execution pipeline

├── chatbot\_ui.py              # Streamlit UI (run this)

├── requirements.txt

├── .env

└── .gitignore



🚀 Quick Start

1\. Create and activate environment

python -m venv venv

source venv/bin/activate   # Windows: venv\\Scripts\\activate



2\. Install dependencies

pip install -r requirements.txt



3\. Configure environment



Create a .env file:



OPENAI\_API\_KEY=your\_key\_here

PINECONE\_API\_KEY=your\_key

PINECONE\_ENV=your\_env





(Additional config options live in config/settings.py.)



▶️ Running the System

Start the chatbot UI

streamlit run chatbot\_ui.py



What happens next:



User is asked for a user ID



System checks if the user already exists



If new → profile initialization



Conversation begins



Memory, verification, and retrieval all run automatically



You interact with the agent like a normal chat — but internally it’s executing a full reasoning pipeline.



🧠 How the Agent Thinks (High Level)

1\. Meta Agent



Decides how to answer:



Simple → fast path



Complex → full reasoning chain



High risk → verification enforced



2\. Planner



Breaks the query into structured steps and intents.



3\. Thinker



Retrieves knowledge only when needed, compresses it, and forms a draft answer.



4\. Verifier



Checks:



factual correctness



numerical validity



compliance constraints



If anything fails → loop back.



5\. Explainer



Produces the final answer with:



concise reasoning



inline citations



zero chain-of-thought leakage



🧠 Memory System



The agent remembers:



user preferences



risk tolerance



explanation depth



prior misunderstandings



Memory is:



summarized



token-bounded



scoped per user



This allows long-term personalization without bloating context windows.



🔍 Evaluation \& Safety



Built-in evaluation includes:



Faithfulness checks



Retrieval accuracy



Domain compliance



Numerical consistency



These run offline or periodically and do not affect latency.



🧪 Why This Architecture Works



No monolithic prompts



No hallucination-by-default



No uncontrolled tool calls



No wasted tokens



No blind trust in LLM output



You get predictable behavior, auditable reasoning, and scalable intelligence.



🧭 Final Note



This isn’t a chatbot.

It’s a reasoning system with guardrails.



If you extend it carefully, you can build:



finance copilots



research agents



compliance assistants



internal decision engines



All without losing control of logic or cost.

