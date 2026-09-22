# 🚀 LangChain --- Interview-Ready Learning & Project README

> A structured, practical and interview-oriented LangChain learning
> track covering LLM fundamentals, prompting, LCEL, RAG, agents, memory,
> middleware and production concepts.

------------------------------------------------------------------------

## 🎯 What This Repository Covers

``` text
LLM Foundations
       ↓
LangChain Fundamentals
       ↓
Prompt Engineering
       ↓
Runnables + LCEL
       ↓
Chains
       ↓
Document Ingestion
       ↓
Embeddings + Vector Stores
       ↓
RAG
       ↓
Tools
       ↓
Agents
       ↓
Conversational Memory
       ↓
Agent State + Checkpointing
       ↓
Middleware
       ↓
Production-Ready Agentic AI
```

------------------------------------------------------------------------

# 🧠 1. LLM & Generative AI Foundations

### AI

Artificial Intelligence enables machines to perform tasks that normally
require human intelligence.

### Generative AI

Generative AI produces new content such as text, images, code, audio and
video.

### LLM

A Large Language Model learns statistical patterns from large amounts of
data and uses them to understand and generate language.

### Generation parameters

  Parameter     Meaning
  ------------- -------------------------------------------------
  Temperature   Controls randomness
  Top-K         Limits candidate tokens to K choices
  Top-P         Selects tokens from cumulative probability mass

------------------------------------------------------------------------

# 🔄 2. Evolution of Language Models

``` text
N-Gram
  ↓
RNN
  ↓
LSTM
  ↓
Transformer
  ↓
Modern LLMs
```

### Transformer

Transformers use self-attention to capture relationships between tokens
and enable highly parallelizable sequence processing.

------------------------------------------------------------------------

# 🦜 3. What is LangChain?

LangChain is a framework for building applications powered by language
models.

It provides abstractions and integrations for connecting:

``` text
LLMs
Prompts
Chains
Runnables
Tools
Agents
Retrievers
Vector Stores
Memory / State
External Data
```

### LangChain is NOT an LLM

``` text
OpenAI / Anthropic / AWS Bedrock
            ↓
          LLM

LangChain
    ↓
Framework around the LLM
```

------------------------------------------------------------------------

# ✍️ 4. Prompt Engineering

## PromptTemplate

``` python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "Explain {topic} in simple terms."
)

prompt.invoke({"topic": "RAG"})
```

## ChatPromptTemplate

``` python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Explain {topic}.")
])
```

## MessagesPlaceholder

Used to dynamically insert a list of messages.

``` python
MessagesPlaceholder("history")
```

Think:

> "Put the conversation history here."

## FewShotPromptTemplate

Used when examples demonstrate the expected input-output pattern.

Important components:

-   `examples`
-   `example_prompt`
-   `prefix`
-   `suffix`
-   `input_variables`

------------------------------------------------------------------------

# 🧪 5. Advanced Prompting

### Zero-shot

No examples.

``` text
Instruction → Answer
```

### Zero-shot Chain-of-Thought

No examples + explicitly request step-by-step reasoning.

### Few-shot

Provide examples.

### Few-shot CoT

Provide examples containing reasoning.

### Self-consistency

Generate multiple reasoning paths and aggregate their answers.

``` text
Problem
 ├── Reasoning path 1 → Answer A
 ├── Reasoning path 2 → Answer A
 └── Reasoning path 3 → Answer B

        ↓

Most consistent answer
```

------------------------------------------------------------------------

# 🎛️ 6. Prompt Tuning

Prompt tuning is a parameter-efficient adaptation technique.

The pretrained model is frozen and trainable soft prompt embeddings are
learned.

``` text
LLM weights → Frozen
Soft prompt → Trainable
```

  Technique            Model weights updated?
  -------------------- ------------------------
  Prompt Engineering   ❌
  Prompt Tuning        ❌
  Fine-Tuning          ✅

------------------------------------------------------------------------

# ⚙️ 7. Runnables & LCEL

A Runnable is a fundamental LangChain unit of execution.

``` python
runnable.invoke(...)
runnable.batch(...)
runnable.stream(...)
```

LCEL = LangChain Expression Language.

``` python
chain = prompt | model | parser
```

The pipe means:

``` text
output of A → input of B
```

### Interview hierarchy

``` text
Runnable = executable unit

Chain = composition of Runnables

LCEL = syntax for composition
```

A chain created with LCEL is itself a Runnable.

------------------------------------------------------------------------

# 📚 8. Document Processing

A LangChain `Document` generally contains:

``` python
Document(
    page_content="...",
    metadata={"source": "..."}
)
```

### Document Loader

Converts different sources into standardized `Document` objects.

Examples:

``` text
PDF → PyPDFLoader
CSV → CSVLoader
Web → WebBaseLoader
YouTube → YouTubeLoader
```

------------------------------------------------------------------------

# 📥 9. Data Ingestion

Typical ingestion pipeline:

``` text
Source
  ↓
Load
  ↓
Clean
  ↓
Split
  ↓
Embed
  ↓
Store
```

> RAG ingestion normally does not train the LLM. Processed information
> is stored externally and retrieved at inference time.

------------------------------------------------------------------------

# ✂️ 10. Text Splitting

``` python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
```

### Key concepts

**chunk_size** → approximate maximum chunk size.

**chunk_overlap** → repeated content between neighboring chunks to
preserve context.

``` text
Chunk 1: A B C D E
Chunk 2:         D E F G H
                ↑
             overlap
```

------------------------------------------------------------------------

# 🧮 11. Embeddings

Embeddings convert text into numerical vectors that capture semantic
meaning.

``` text
"Python programming"
        ↓
[0.12, -0.31, 0.82, ...]
```

Similar meanings tend to have nearby vectors.

------------------------------------------------------------------------

# 🗄️ 12. Vector Stores & Retrievers

Example:

``` python
from langchain_chroma import Chroma

db = Chroma.from_documents(
    documents,
    embeddings
)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
```

### Key ideas

-   Vector Store → stores/searches vectors
-   Retriever → provides a retrieval interface
-   `k` → number of relevant chunks/documents returned
-   `as_retriever()` → converts a vector store into a Retriever
    interface

------------------------------------------------------------------------

# 🤖 13. RAG

RAG = Retrieval-Augmented Generation.

``` text
             Documents
                 ↓
              Loader
                 ↓
              Chunks
                 ↓
             Embeddings
                 ↓
             Vector DB
                 ↓
              Retriever
                 ↑
                 │
User Question ───┘
                 ↓
         Relevant Context
                 ↓
                LLM
                 ↓
               Answer
```

> Retrieve relevant external information and provide it to the LLM as
> context before generating the answer.

------------------------------------------------------------------------

# 📦 14. Stuff Documents Chain

``` python
docs_chain = create_stuff_documents_chain(
    llm,
    prompt
)
```

Its responsibility:

> Combine retrieved documents into a context variable and pass that
> context to the LLM.

``` text
Documents
    ↓
Combine
    ↓
{context}
    ↓
Prompt
    ↓
LLM
```

------------------------------------------------------------------------

# 🔍 15. Retrieval Chain

``` python
retriever_chain = create_retrieval_chain(
    retriever,
    docs_chain
)
```

``` text
Question
   ↓
Retriever
   ↓
Relevant documents
   ↓
Stuff Documents Chain
   ↓
LLM
   ↓
Answer
```

### Interview distinction

> Retrieval Chain decides **which documents to retrieve**; Stuff
> Documents Chain decides **how to provide those documents to the LLM**.

------------------------------------------------------------------------

# 🛠️ 16. Tools

A tool gives an agent an external capability.

``` python
from langchain_core.tools import tool

@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
```

Type hints and docstrings help define and describe the tool.

------------------------------------------------------------------------

# 🤖 17. Agents

An agent uses an LLM to dynamically decide which tools to use and in
what sequence.

``` text
User
 ↓
LLM
 ↓
Choose tool
 ↓
Execute tool
 ↓
Tool result
 ↓
LLM
 ↓
Another tool / Final answer
```

### Core distinction

``` text
LLM   = reasoning model
Agent = orchestration/decision system
Tool  = external capability
```

The LLM produces a structured tool-call request; the agent runtime
executes the tool.

### Chain vs Agent

``` text
CHAIN
Developer defines sequence.

A → B → C


AGENT
LLM dynamically decides.

A → ? → ? → C
```

------------------------------------------------------------------------

# 🧰 18. Agent Invocation

Modern agent APIs commonly use message-based state.

``` python
agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Search for the latest news."
        }
    ]
})
```

### Structure

``` text
invoke()
   ↓
dict/state
   ↓
messages
   ↓
list
   ↓
message
   ├── role
   └── content
```

------------------------------------------------------------------------

# 🧠 19. Conversational Memory

Memory allows applications to use information from previous
interactions.

``` text
Previous messages
       ↓
Prompt
       ↓
LLM
       ↓
Context-aware response
```

------------------------------------------------------------------------

# 💬 20. RunnableWithMessageHistory

Used mainly for chain-based conversational memory.

``` python
RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)
```

Think:

> **RunnableWithMessageHistory = manages conversation history around a
> Runnable.**

------------------------------------------------------------------------

# 🗃️ 21. InMemoryChatMessageHistory

Stores messages in the application's RAM.

Conceptually:

``` python
store = {
    "user123": InMemoryChatMessageHistory()
}
```

The history contains:

``` text
HumanMessage
AIMessage
HumanMessage
AIMessage
```

It is temporary. Restarting the application clears in-memory state.

------------------------------------------------------------------------

# 🆔 22. session_id

Identifies which conversation history should be used.

``` python
config = {
    "configurable": {
        "session_id": "test_session"
    }
}
```

Think:

``` text
session_id
    ↓
Which conversation?
    ↓
Chat history
```

------------------------------------------------------------------------

# 💾 23. InMemorySaver & Checkpointing

`InMemorySaver` is a checkpointer used with stateful agents/LangGraph
workflows.

``` python
memory = InMemorySaver()

agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=memory
)
```

It stores workflow/agent state as checkpoints in memory.

> It does not mean "store the agent's private thoughts." It stores
> checkpointed workflow state.

------------------------------------------------------------------------

# 🧵 24. thread_id

Used to identify a stateful agent/graph execution thread.

``` python
config = {
    "configurable": {
        "thread_id": "Mar26"
    }
}
```

Think:

``` text
thread_id
    ↓
Which agent workflow?
    ↓
Checkpointed state
```

------------------------------------------------------------------------

# session_id vs thread_id

                    `session_id`                   `thread_id`
  ----------------- ------------------------------ ----------------------------
  Common use        Chain memory                   Agent/LangGraph state
  Identifies        Conversation                   Workflow/state thread
  Storage example   `InMemoryChatMessageHistory`   `InMemorySaver`
  Main focus        Messages                       Complete state/checkpoints

> `session_id` → **Which conversation?**\
> `thread_id` → **Which agent workflow?**

------------------------------------------------------------------------

# 🧩 25. Conversational RAG

RAG + memory:

``` text
User Question
      │
      ├──────────────→ Retriever
      │                     ↓
      │               Relevant Docs
      │                     ↓
      │              Stuff Documents
      │                     ↓
      ↓                     ↓
Chat History ───────────→ Prompt
                            ↓
                           LLM
                            ↓
                          Answer
                            ↓
                       Save History
```

This is **Conversational RAG**.

------------------------------------------------------------------------

# 🛡️ 26. Middleware

Middleware is a control layer around agent execution.

``` text
User
 ↓
Middleware
 ↓
Agent
 ↓
Middleware
 ↓
Response
```

It can intercept, inspect, modify or control execution.

### Middleware vs Guardrails

> **Middleware = broad control mechanism.**\
> **Guardrails = rules/controls intended to prevent unwanted or unsafe
> behavior.**

Guardrails can be implemented through middleware, but the concepts are
not identical.

------------------------------------------------------------------------

# 🧱 27. Prebuilt Middleware

## Summarization Middleware

Controls long conversations.

``` text
Long history
     ↓
Summarize old messages
     +
Keep recent messages
     ↓
Smaller context
```

Purpose:

-   Prevent context overflow
-   Reduce token usage
-   Preserve important context
-   Maintain performance

## PII Detection Middleware

Detects Personally Identifiable Information.

Examples:

-   Phone numbers
-   Email addresses
-   Government identifiers
-   Financial information

## Model Call Limit Middleware

Controls the number of LLM/model calls.

``` text
Maximum model calls = N
```

Useful for cost, latency and preventing uncontrolled loops.

## Tool Call Limit Middleware

Controls how many times an agent can execute tools.

``` text
Model Call Limit → How many times can the LLM run?
Tool Call Limit  → How many times can tools run?
```

## Human-in-the-Loop Middleware

Pauses execution for human approval.

``` text
Agent wants sensitive action
             ↓
           PAUSE
             ↓
      Human approval
        ↙       ↘
      YES        NO
       ↓          ↓
   Continue      Stop
```

------------------------------------------------------------------------

# 🏗️ 28. End-to-End Agent Architecture

``` text
                         USER
                           │
                           ▼
                    ┌─────────────┐
                    │ Middleware  │
                    └──────┬──────┘
                           │
                           ▼
                       AGENT
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
            LLM          Tools        Memory
              │            │            │
              │            │            ▼
              │            │       Checkpointer
              │            │
              ▼            ▼
        Reasoning       External
                         Systems
              │
              ▼
           Response
```

------------------------------------------------------------------------

# 💡 29. High-Value Interview Distinctions

  -----------------------------------------------------------------------
  Concept                             Remember
  ----------------------------------- -----------------------------------
  LLM vs LangChain                    Model vs application framework

  Chain vs Agent                      Fixed workflow vs dynamic workflow

  Tool vs Agent                       Capability vs
                                      decision/orchestration

  Retriever vs Vector Store           Retrieval interface vs vector
                                      storage

  RAG vs Fine-tuning                  External context vs model parameter
                                      updates

  Memory vs RAG                       Conversation state vs external
                                      knowledge

  RunnableWithMessageHistory vs       Message history vs workflow
  InMemorySaver                       checkpoints

  session_id vs thread_id             Conversation identity vs
                                      agent/graph state identity

  Middleware vs Tool                  Control layer vs capability
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 🧪 30. Core RAG Pattern

``` python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

loader = PyPDFLoader("document.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

docs = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma.from_documents(
    docs,
    embeddings
)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
```

------------------------------------------------------------------------

# 🤖 31. Core Agent Pattern

``` python
from langchain_core.tools import tool
from langchain.agents import create_agent

@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

agent = create_agent(
    model=llm,
    tools=[add_numbers],
    system_prompt="You are a helpful assistant."
)

response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Add 10 and 20."
        }
    ]
})
```

------------------------------------------------------------------------

# 🧠 32. Core Memory Pattern

``` python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]

conversation = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

config = {
    "configurable": {
        "session_id": "user123"
    }
}

conversation.invoke(
    {"input": "My name is Ram"},
    config=config
)
```

------------------------------------------------------------------------

# 💼 33. Interview Questions

### Fundamentals

1.  What is LangChain?
2.  Why do we need LangChain if LLMs already exist?
3.  What is a Runnable?
4.  What is LCEL?
5.  Chain vs Agent?

### Prompting

6.  PromptTemplate vs ChatPromptTemplate?
7.  What is MessagesPlaceholder?
8.  Zero-shot vs few-shot?
9.  What is prompt tuning?
10. Prompt tuning vs fine-tuning?

### RAG

11. What is RAG?
12. Why split documents?
13. What is chunk overlap?
14. What are embeddings?
15. What is a vector database?
16. What does `k` mean?
17. Vector store vs retriever?
18. RAG vs fine-tuning?
19. What is Stuff Documents Chain?

### Agents

20. What is an agent?
21. Is the LLM the agent?
22. What is a tool?
23. How does tool calling work?
24. Agent vs chain?
25. Why are tool descriptions important?

### Memory

26. What is conversational memory?
27. What is `RunnableWithMessageHistory`?
28. What is `InMemoryChatMessageHistory`?
29. What is `InMemorySaver`?
30. session_id vs thread_id?
31. What is checkpointing?

### Middleware

32. What is middleware?
33. Middleware vs guardrails?
34. Why use summarization middleware?
35. Model call limit vs tool call limit?
36. What is human-in-the-loop?

------------------------------------------------------------------------

# 🎯 34. Final Mental Model

``` text
                         USER
                           │
                           ▼
                      PROMPT
                           │
                           ▼
                     LANGCHAIN
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
      Runnable          Retriever          Agent
          │                │                │
          ▼                ▼                ▼
        Chain             RAG             Tools
          │                                 │
          │                                 ▼
          │                              APIs/DB
          │
          └──────────────┐
                         ▼
                    LLM / Model
                         │
                         ▼
                  Memory / State
                         │
                         ▼
                    Middleware
                         │
                         ▼
                      Output
```

------------------------------------------------------------------------

# 🚀 35. What Comes After LangChain?

LangChain provides the components. The next level is learning how to
orchestrate stateful agentic workflows.

Recommended progression:

``` text
LangChain ✅
    ↓
LangGraph
    ↓
Stateful Agents
    ↓
Multi-Agent Systems
    ↓
Advanced RAG
    ↓
Structured Outputs
    ↓
Evaluation
    ↓
Observability
    ↓
Production Agentic AI
```

------------------------------------------------------------------------

# 🏁 Study Status

## LangChain Track: ✅ COMPLETED

-   [x] LLM & GenAI fundamentals
-   [x] LangChain fundamentals
-   [x] Prompt engineering
-   [x] Advanced prompting
-   [x] Prompt tuning
-   [x] Runnables
-   [x] LCEL
-   [x] Chains
-   [x] Document loaders
-   [x] Data ingestion
-   [x] Text splitting
-   [x] Embeddings
-   [x] Vector stores
-   [x] Retrievers
-   [x] RAG
-   [x] Document chains
-   [x] Tools
-   [x] Agents
-   [x] Conversational memory
-   [x] Checkpointing
-   [x] Agent state concepts
-   [x] Middleware
-   [x] Summarization middleware
-   [x] PII middleware
-   [x] Model/tool call limits
-   [x] Human-in-the-loop concepts

> ## 🎉 LangChain: COMPLETE
>
> **Next major specialization: LangGraph + Agentic AI Architecture**
