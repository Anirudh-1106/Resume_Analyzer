"""
generate_cse_resumes.py
Generates modern CSE-focused synthetic resumes and appends them to
resume_app/data/structured_resumes.csv.

Domains added (new):
  DATA-SCIENCE, MACHINE-LEARNING, AI-ENGINEERING, DATA-ENGINEERING,
  FRONTEND, FULLSTACK, BACKEND, DEVOPS-CLOUD, CYBERSECURITY,
  MOBILE-DEVELOPMENT, UI-UX
"""

import os
import random
import pandas as pd

random.seed(42)

# ---------------------------------------------------------------------------
# Skill pools per role
# ---------------------------------------------------------------------------

ROLE_DATA = {
    "DATA-SCIENCE": {
        "titles": [
            "Data Scientist", "Senior Data Scientist", "Lead Data Scientist",
            "Applied Scientist", "Research Data Scientist", "ML Data Scientist",
            "Principal Data Scientist", "Data Science Analyst",
        ],
        "core_skills": [
            "python", "machine learning", "deep learning", "pandas", "numpy",
            "scikit-learn", "statistics", "matplotlib", "seaborn", "jupyter",
            "sql", "r", "tensorflow", "pytorch", "keras", "xgboost", "lightgbm",
            "feature engineering", "model evaluation", "a/b testing",
            "hypothesis testing", "regression", "classification", "clustering",
            "dimensionality reduction", "pca", "cross validation",
            "hugging face", "mlflow", "data visualization", "tableau", "power bi",
            "spark", "big data", "nlp", "computer vision", "llm",
        ],
        "optional_skills": [
            "databricks", "snowflake", "airflow", "dbt", "google colab",
            "azure ml", "aws sagemaker", "gcp vertex ai", "optuna", "wandb",
            "shap", "lime", "plotly", "streamlit", "docker", "git",
            "time series", "forecasting", "anomaly detection",
        ],
        "certs": [
            "Google Professional Data Engineer",
            "IBM Data Science Professional Certificate",
            "AWS Certified Machine Learning Specialty",
            "Microsoft Certified: Azure Data Scientist Associate",
            "Coursera Deep Learning Specialization",
            "DataCamp Data Scientist Professional",
            "TensorFlow Developer Certificate",
        ],
        "summaries": [
            "Results-driven Data Scientist with expertise in building predictive models and extracting actionable insights from large-scale datasets. Proficient in Python, ML frameworks, and cloud platforms.",
            "Analytical Data Scientist skilled in statistical modeling, machine learning pipelines, and data storytelling. Strong background in Python, SQL, and deep learning.",
            "Experienced Data Scientist specializing in NLP, computer vision, and time-series forecasting. Expert in deploying ML models to production using MLflow and cloud services.",
            "Data Scientist with 4+ years transforming raw data into business value using advanced ML algorithms, A/B testing, and interactive dashboards using Tableau and Power BI.",
            "Innovative Data Scientist passionate about LLMs, generative AI, and building RAG pipelines. Proficient in Hugging Face Transformers, LangChain, and vector databases.",
        ],
        "projects": [
            "Customer Churn Prediction using XGBoost with 93% accuracy and SHAP explainability dashboard.",
            "Sentiment Analysis pipeline using BERT fine-tuned on 500k customer reviews achieving F1 of 0.91.",
            "Real-time fraud detection system processing 10M transactions/day using streaming Spark and RandomForest.",
            "Sales forecasting model using LSTM reducing forecast error by 38% vs baseline ARIMA.",
            "Recommendation engine using collaborative filtering and matrix factorization for e-commerce platform.",
            "LLM-powered document Q&A system using RAG, LangChain, and Pinecone vector database.",
            "Computer vision model for defect detection in manufacturing with 97% precision using YOLOv8.",
            "COVID-19 spread prediction dashboard using epidemiological ML models and Streamlit.",
        ],
        "internships": [
            "Data Science Intern at Google: Developed A/B testing framework reducing model deployment time by 20%.",
            "ML Research Intern at Microsoft Research: Contributed to NLP model for document summarization.",
            "Data Science Intern at Flipkart: Built product recommendation engine improving CTR by 15%.",
            "Analytics Intern at Accenture: Designed automated reporting pipeline saving 10 hours/week.",
            "AI Intern at Infosys: Developed chatbot using RASA and integrated with enterprise CRM.",
        ],
    },

    "MACHINE-LEARNING": {
        "titles": [
            "ML Engineer", "Machine Learning Engineer", "Senior ML Engineer",
            "MLOps Engineer", "ML Platform Engineer", "Applied ML Engineer",
            "ML Infrastructure Engineer", "Production ML Engineer",
        ],
        "core_skills": [
            "python", "machine learning", "tensorflow", "pytorch", "scikit-learn",
            "mlflow", "kubeflow", "docker", "kubernetes", "fastapi",
            "feature engineering", "model deployment", "model monitoring",
            "ci/cd", "aws sagemaker", "azure ml", "gcp vertex ai",
            "apache kafka", "spark", "sql", "numpy", "pandas",
            "deep learning", "neural networks", "hyperparameter tuning",
            "onnx", "triton inference server", "bentoml", "seldon",
            "a/b testing", "shadow deployment", "canary release",
        ],
        "optional_skills": [
            "rust", "go", "c++", "java", "grpc", "protobuf", "redis",
            "airflow", "prefect", "great expectations", "dvc", "wandb",
            "optuna", "ray tune", "terraform", "helm", "argocd",
            "prometheus", "grafana", "elastic search",
        ],
        "certs": [
            "AWS Certified Machine Learning Specialty",
            "Google Professional Machine Learning Engineer",
            "Microsoft Certified: Azure AI Engineer Associate",
            "MLOps Specialization – Coursera",
            "TensorFlow Developer Certificate",
            "Kubeflow MLOps Professional",
        ],
        "summaries": [
            "MLOps-focused Machine Learning Engineer with expertise in building, deploying, and monitoring production ML systems at scale using Kubernetes, MLflow, and cloud platforms.",
            "Machine Learning Engineer specializing in model optimization, feature stores, and ML pipelines. Proficient in PyTorch, FastAPI, and AWS SageMaker.",
            "Senior ML Engineer with experience designing end-to-end ML platforms, real-time inference systems, and automated retraining pipelines for high-traffic applications.",
            "ML Engineer passionate about bringing research models to production reliably. Expert in Docker, Kubernetes, CI/CD pipelines, and model monitoring with Prometheus.",
            "Applied ML Engineer with strong background in recommendation systems, real-time scoring APIs, and distributed training using Spark and Horovod.",
        ],
        "projects": [
            "Built end-to-end MLOps pipeline on AWS using SageMaker Pipelines, MLflow tracking, and automated retraining triggers based on data drift.",
            "Designed real-time model serving system on Kubernetes with Triton Inference Server handling 50k req/sec with P99 latency <10ms.",
            "Developed feature store using Feast integrated with Kafka and Redis for real-time + batch feature serving.",
            "Implemented canary deployment framework for ML models reducing rollback time from hours to minutes.",
            "GPU-optimized training pipeline for LLM fine-tuning using DeepSpeed ZeRO and gradient checkpointing.",
            "AutoML system using Optuna for hyperparameter search across 50+ model configurations.",
        ],
        "internships": [
            "ML Engineering Intern at Amazon: Built automated model evaluation harness reducing release cycle by 30%.",
            "MLOps Intern at Uber: Contributed to internal feature store improving feature reuse across 20+ teams.",
            "ML Platform Intern at LinkedIn: Developed model monitoring dashboard detecting feature drift in real-time.",
        ],
    },

    "AI-ENGINEERING": {
        "titles": [
            "AI Engineer", "NLP Engineer", "Generative AI Engineer",
            "LLM Engineer", "Conversational AI Engineer", "AI Research Engineer",
            "Computer Vision Engineer", "AI Solutions Architect",
        ],
        "core_skills": [
            "python", "nlp", "large language models", "langchain", "llama index",
            "hugging face", "transformers", "bert", "gpt", "openai api",
            "prompt engineering", "rag", "vector databases", "pinecone",
            "chroma", "weaviate", "spacy", "nltk", "fine-tuning",
            "lora", "qlora", "peft", "reinforcement learning from human feedback",
            "computer vision", "opencv", "yolo", "stable diffusion",
            "multimodal models", "embeddings", "semantic search",
            "pytorch", "tensorflow", "cuda", "gpu programming",
            "fastapi", "streamlit", "gradio",
        ],
        "optional_skills": [
            "anthropic api", "google gemini", "azure openai", "mistral",
            "llama", "ollama", "vllm", "text generation webui",
            "knowledge graphs", "neo4j", "memory management",
            "agent frameworks", "autogen", "crewai", "semantic kernel",
            "diffusers", "controlnet", "whisper", "deep speech",
        ],
        "certs": [
            "DeepLearning.AI LLM Specialization",
            "Microsoft Certified: Azure AI Engineer Associate",
            "AWS Certified AI Practitioner",
            "Google Cloud Professional ML Engineer",
            "Hugging Face NLP Course Certificate",
            "NVIDIA Deep Learning Institute Certificate",
        ],
        "summaries": [
            "Generative AI Engineer with expertise in LLM fine-tuning, RAG pipelines, and AI agent systems. Built production-grade applications using LangChain, OpenAI, and vector databases.",
            "NLP Engineer specializing in transformer models, conversational AI, and semantic search. Proficient in Hugging Face, spaCy, and multi-modal LLM applications.",
            "AI Engineer with deep expertise in computer vision, object detection, and generative models. Experienced in deploying YOLO, Stable Diffusion, and Vision Transformers.",
            "LLM Engineer focused on building intelligent agent frameworks, RAG systems, and enterprise AI assistants. Expert in LangChain, LlamaIndex, and prompt engineering.",
            "Conversational AI Engineer with 3+ years building chatbots, voice assistants, and intent recognition systems for enterprise applications.",
        ],
        "projects": [
            "Enterprise RAG chatbot using LangChain + Pinecone + GPT-4 answering queries over 10,000-page knowledge base with 89% accuracy.",
            "Fine-tuned LLaMA-3 on domain-specific legal dataset using QLoRA achieving 40% improvement on legal Q&A benchmarks.",
            "Multi-agent AI research assistant using AutoGen coordinating 5 specialized agents for autonomous research tasks.",
            "Real-time speech-to-text + sentiment analysis pipeline using Whisper and fine-tuned BERT for call center analytics.",
            "Text-to-image generation system using Stable Diffusion with ControlNet for product visualization in e-commerce.",
            "Semantic search engine over 1M product catalog using bi-encoder embeddings and FAISS achieving 95% Recall@10.",
            "AI code review assistant powered by GPT-4 Turbo integrated into GitHub PR workflow reducing review time by 40%.",
        ],
        "internships": [
            "AI Research Intern at OpenAI: Contributed to RLHF data pipeline and preference modeling experiments.",
            "NLP Intern at Samsung Research: Built intent classification model for voice assistant with 96% accuracy.",
            "Computer Vision Intern at Tesla: Developed object tracking module for autonomous driving perception stack.",
            "AI Engineering Intern at Wipro: Created document extraction system using LayoutLM reducing manual effort by 70%.",
        ],
    },

    "DATA-ENGINEERING": {
        "titles": [
            "Data Engineer", "Senior Data Engineer", "Data Platform Engineer",
            "Analytics Engineer", "Big Data Engineer", "ETL Developer",
            "Data Infrastructure Engineer", "Cloud Data Engineer",
        ],
        "core_skills": [
            "python", "sql", "apache spark", "apache kafka", "apache airflow",
            "dbt", "hadoop", "hive", "aws", "azure", "gcp", "databricks",
            "snowflake", "bigquery", "redshift", "postgresql", "mysql",
            "mongodb", "redis", "elasticsearch", "data pipelines", "etl",
            "elt", "data warehousing", "data modeling", "star schema",
            "dimensional modeling", "delta lake", "apache flink",
            "docker", "kubernetes", "terraform", "git", "ci/cd",
            "data quality", "great expectations", "data catalog",
        ],
        "optional_skills": [
            "scala", "java", "go", "pulsar", "nats", "iceberg", "hudi",
            "trino", "presto", "duckdb", "polars", "ray", "dask",
            "fivetran", "stitch", "mage", "prefect", "dagster",
            "azure data factory", "aws glue", "gcp dataflow",
            "looker", "metabase", "apache superset",
        ],
        "certs": [
            "Google Professional Data Engineer",
            "AWS Certified Data Analytics Specialty",
            "Microsoft Certified: Azure Data Engineer Associate",
            "Databricks Certified Associate Developer for Apache Spark",
            "Snowflake Pro Core Certification",
            "dbt Analytics Engineering Certification",
        ],
        "summaries": [
            "Data Engineer with expertise in building scalable data pipelines, data warehouses, and lakehouse architectures using Spark, Databricks, and cloud platforms.",
            "Senior Data Engineer specializing in real-time streaming systems with Kafka, Flink, and Delta Lake. Strong background in data modeling and ELT transformations with dbt.",
            "Cloud Data Engineer experienced in migrating on-premise data infrastructure to AWS/Azure data platforms. Expert in Airflow orchestration and Snowflake optimization.",
            "Analytics Engineer bridging data engineering and analytics — building dbt models, semantic layers, and self-serve analytics platforms for business intelligence teams.",
            "Big Data Engineer skilled in processing petabyte-scale datasets using distributed computing (Spark, Hadoop) and building fault-tolerant streaming pipelines.",
        ],
        "projects": [
            "Real-time data lakehouse on Azure using Kafka + Databricks + Delta Lake processing 5TB/day from 200+ microservices.",
            "Built dbt transformation layer over Snowflake with 300+ models, automated testing, and CI/CD via GitHub Actions.",
            "Event-driven data pipeline using Kafka Streams for financial transaction processing with exactly-once semantics.",
            "Multi-cloud data platform migration (on-prem Oracle → AWS Redshift + S3) for retail client with zero downtime.",
            "Data quality monitoring framework using Great Expectations integrated into Airflow DAGs with Slack alerting.",
            "Streaming analytics pipeline using Apache Flink processing IoT sensor data from 10,000 devices in real-time.",
        ],
        "internships": [
            "Data Engineering Intern at Swiggy: Built ETL pipeline ingesting 50M daily order events into BigQuery.",
            "Data Platform Intern at Razorpay: Developed dbt models for financial reporting reducing reconciliation time by 60%.",
            "Big Data Intern at TCS: Optimized Spark jobs processing 2TB daily log files reducing runtime from 4hrs to 40min.",
        ],
    },

    "FRONTEND": {
        "titles": [
            "Frontend Developer", "Senior Frontend Engineer", "React Developer",
            "UI Developer", "Frontend Architect", "Web Developer",
            "JavaScript Engineer", "Next.js Developer",
        ],
        "core_skills": [
            "javascript", "typescript", "react", "next.js", "html5", "css3",
            "tailwind css", "sass", "redux", "zustand", "react query",
            "graphql", "rest api", "webpack", "vite", "jest",
            "react testing library", "cypress", "storybook", "figma",
            "responsive design", "web accessibility", "wcag", "seo",
            "performance optimization", "core web vitals", "lighthouse",
            "git", "npm", "yarn", "pnpm",
        ],
        "optional_skills": [
            "vue.js", "angular", "svelte", "astro", "remix",
            "three.js", "d3.js", "webgl", "web animations api",
            "pwa", "service workers", "web workers", "webassembly",
            "react native", "electron", "tauri",
            "micro frontends", "module federation", "turborepo",
            "framer motion", "gsap", "adobe xd",
        ],
        "certs": [
            "Meta Front-End Developer Professional Certificate",
            "Google UX Design Certificate",
            "AWS Certified Cloud Practitioner",
            "Microsoft Certified: Azure Developer Associate",
            "Udemy React - The Complete Guide Certificate",
            "Frontend Masters JavaScript Performance Certificate",
        ],
        "summaries": [
            "Frontend Developer with 3+ years building performant, accessible React applications. Expert in TypeScript, Next.js, and Tailwind CSS with a passion for great user experiences.",
            "Senior Frontend Engineer specializing in large-scale React SPAs, micro-frontend architectures, and design system development. Strong focus on performance and accessibility.",
            "Creative Frontend Developer with expertise in React animations, WebGL, and interactive data visualizations using D3.js and Three.js.",
            "Next.js Developer experienced in server-side rendering, static site generation, and full-stack React applications with API routes and edge middleware.",
            "Frontend Architect with expertise in module federation, monorepo setups with Turborepo, and cross-team component library governance.",
        ],
        "projects": [
            "E-commerce platform built with Next.js 14, achieving Core Web Vitals score >95 with ISR, image optimization, and edge caching.",
            "Design system with 80+ React components, Storybook documentation, visual regression testing, and automated NPM publishing via CI/CD.",
            "Real-time collaborative whiteboard using React, WebSocket, and Canvas API supporting 100+ concurrent users.",
            "Dashboard built with React + D3.js visualizing 1M+ data points with WebWorker-offloaded aggregations for smooth 60fps rendering.",
            "Progressive Web App with offline support using Service Workers, background sync, and push notifications achieving 100 Lighthouse score.",
            "Micro-frontend architecture using Module Federation across 5 independent React apps deployed on Netlify edge.",
        ],
        "internships": [
            "Frontend Intern at Razorpay: Built responsive payment UI components in React reducing checkout abandonment by 12%.",
            "React Developer Intern at OYO: Developed hotel listing page improving LCP by 40% through lazy loading and image CDN.",
            "Web Development Intern at Internshala: Created interactive dashboard using React and Chart.js for 1M+ users.",
        ],
    },

    "FULLSTACK": {
        "titles": [
            "Full Stack Developer", "Senior Full Stack Engineer",
            "Full Stack Engineer", "Software Engineer (Full Stack)",
            "MERN Stack Developer", "MEAN Stack Developer",
            "Full Stack Web Developer", "Full Stack Architect",
        ],
        "core_skills": [
            "javascript", "typescript", "react", "next.js", "node.js",
            "express.js", "python", "django", "fastapi", "postgresql",
            "mysql", "mongodb", "redis", "rest api", "graphql",
            "docker", "aws", "ci/cd", "git", "html5", "css3",
            "tailwind css", "jest", "pytest", "sql", "microservices",
            "websockets", "authentication", "jwt", "oauth2",
            "nginx", "linux", "bash scripting",
        ],
        "optional_skills": [
            "kubernetes", "terraform", "ansible", "grpc", "kafka",
            "elastic search", "vue.js", "angular", "svelte",
            "go", "rust", "java", "spring boot", "celery",
            "prisma", "drizzle", "typeorm", "sequelize",
            "stripe api", "twilio", "sendgrid", "firebase",
        ],
        "certs": [
            "AWS Certified Developer Associate",
            "MongoDB Certified Developer",
            "Meta Full Stack Developer Professional Certificate",
            "Google Associate Cloud Engineer",
            "Node.js Application Developer (IBM)",
            "Docker Certified Associate",
        ],
        "summaries": [
            "Full Stack Developer with 4+ years building end-to-end web applications using React, Node.js, and PostgreSQL. Experienced in microservices, cloud deployment, and Agile workflows.",
            "MERN Stack Engineer skilled in building scalable APIs, real-time features, and responsive frontends. Strong DevOps background with Docker and AWS.",
            "Senior Full Stack Developer specializing in Next.js, FastAPI, and PostgreSQL. Experience leading teams of 5+ engineers delivering complex SaaS products.",
            "Full Stack Engineer with expertise in building multi-tenant SaaS platforms with role-based access control, payment integration, and real-time collaboration features.",
            "Creative full stack developer with a product mindset, building MVPs to production-grade applications using modern JavaScript/TypeScript ecosystem.",
        ],
        "projects": [
            "SaaS project management tool (Jira alternative) built with Next.js, FastAPI, PostgreSQL, WebSocket for real-time updates — serving 10k+ users.",
            "E-learning platform with video streaming, live sessions, quiz engine, and payment integration (Stripe) built on MERN stack.",
            "Real-time food delivery tracking app with WebSocket, Google Maps API, React Native mobile frontend, and Node.js backend.",
            "Multi-tenant CRM system with React frontend, Django REST Framework, PostgreSQL, and Redis caching serving 500+ clients.",
            "Open-source note-taking app (Notion alternative) using Next.js, Tiptap editor, and Supabase with real-time collaboration.",
        ],
        "internships": [
            "Full Stack Intern at Zoho: Developed internal HR portal using React + Java Spring Boot serving 5,000 employees.",
            "Software Engineering Intern at Freshworks: Built bulk email campaign feature in Rails + React with SendGrid integration.",
            "Full Stack Intern at Startupathon: Created MVP for HealthTech startup using MERN stack in 6 weeks.",
        ],
    },

    "BACKEND": {
        "titles": [
            "Backend Developer", "Backend Engineer", "Senior Backend Engineer",
            "API Developer", "Python Backend Developer", "Node.js Developer",
            "Java Backend Engineer", "Go Developer",
        ],
        "core_skills": [
            "python", "node.js", "java", "go", "sql", "postgresql",
            "mysql", "mongodb", "redis", "rest api", "graphql", "grpc",
            "microservices", "docker", "kubernetes", "aws", "linux",
            "fastapi", "django", "express.js", "spring boot",
            "message queues", "rabbitmq", "kafka", "celery",
            "authentication", "jwt", "oauth2", "api gateway",
            "nginx", "load balancing", "caching", "ci/cd", "git",
            "unit testing", "integration testing", "pytest", "junit",
        ],
        "optional_skills": [
            "rust", "c++", "scala", "elixir", "phoenix",
            "elastic search", "cassandra", "dynamodb", "firebase",
            "terraform", "ansible", "helm", "istio",
            "websockets", "server-sent events", "webhooks",
            "openapi", "swagger", "postman", "datadog", "newrelic",
        ],
        "certs": [
            "AWS Certified Developer Associate",
            "Google Associate Cloud Engineer",
            "Microsoft Certified: Azure Developer Associate",
            "Oracle Java SE Programmer Certification",
            "MongoDB Certified Developer",
            "Docker Certified Associate",
        ],
        "summaries": [
            "Backend Engineer with 4+ years designing high-performance REST and gRPC APIs using Python, Go, and PostgreSQL. Expert in microservices, event-driven architecture, and cloud deployment.",
            "Senior Backend Developer specializing in distributed systems, database optimization, and API design. Proficient in Java Spring Boot and Kubernetes.",
            "Python backend engineer with deep FastAPI/Django expertise building scalable APIs serving millions of requests. Strong database optimization and caching skills.",
            "Node.js Backend Developer with expertise in real-time systems using WebSockets and event-driven architecture. Experience with GraphQL federation and microservices.",
            "Go Backend Engineer focused on high-throughput, low-latency services. Experience building fintech payment APIs processing $1M+ daily transactions.",
        ],
        "projects": [
            "Distributed payments API using Go + PostgreSQL with idempotency, retry logic, and end-to-end encryption processing 100k TPS.",
            "Event-driven microservices platform using Java Spring Boot, Kafka, and Docker Compose orchestrating 12 services.",
            "GraphQL federation gateway in Node.js unifying 8 downstream microservices with DataLoader batching and Redis caching.",
            "Real-time notification service using WebSockets + Redis Pub/Sub supporting 500k concurrent connections.",
            "High-performance search API using Elasticsearch + FastAPI with custom synonym filters and relevance tuning.",
            "REST API gateway with rate limiting, circuit breakers, JWT auth, and OpenTelemetry distributed tracing.",
        ],
        "internships": [
            "Backend Intern at CRED: Developed rewards calculation engine in Python handling 2M events/day.",
            "API Development Intern at Paytm: Built UPI transaction status polling service with exponential backoff.",
            "Backend Engineering Intern at Juspay: Implemented idempotency layer for payment APIs reducing duplicate charges by 99%.",
        ],
    },

    "DEVOPS-CLOUD": {
        "titles": [
            "DevOps Engineer", "Cloud Engineer", "Site Reliability Engineer",
            "Platform Engineer", "Infrastructure Engineer",
            "DevSecOps Engineer", "Cloud Architect",
            "Kubernetes Engineer", "SRE Lead",
        ],
        "core_skills": [
            "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
            "ansible", "helm", "ci/cd", "jenkins", "github actions",
            "gitlab ci", "argocd", "gitlab", "linux", "bash",
            "python", "go", "prometheus", "grafana", "datadog",
            "elastic stack", "logging", "monitoring", "alerting",
            "networking", "vpc", "load balancers", "dns", "ssl/tls",
            "security", "iam", "secrets management", "vault",
            "infrastructure as code", "gitops", "service mesh", "istio",
        ],
        "optional_skills": [
            "pulumi", "crossplane", "flux", "tekton", "spinnaker",
            "opentelemetry", "jaeger", "zipkin", "chaos engineering",
            "litmus chaos", "cost optimization", "finops",
            "ebpf", "cilium", "envoy", "nginx", "haproxy",
            "cloudflare", "fastly", "waf", "ddos protection",
        ],
        "certs": [
            "AWS Certified DevOps Engineer Professional",
            "Certified Kubernetes Administrator (CKA)",
            "Certified Kubernetes Application Developer (CKAD)",
            "HashiCorp Certified: Terraform Associate",
            "Google Professional DevOps Engineer",
            "Microsoft Certified: DevOps Engineer Expert",
            "Docker Certified Associate",
            "AWS Certified Solutions Architect Associate",
        ],
        "summaries": [
            "DevOps Engineer with expertise in Kubernetes, Terraform, and GitOps workflows. Experienced in building robust CI/CD pipelines and cloud infrastructure for high-availability systems.",
            "Site Reliability Engineer focused on achieving 99.99% uptime through SLO/SLA management, automated incident response, and chaos engineering.",
            "Cloud Architect with deep AWS expertise in designing multi-region, fault-tolerant architectures using EKS, RDS, CloudFront, and serverless technologies.",
            "Platform Engineer building internal developer platforms (IDPs) using Backstage, Kubernetes, and GitOps to improve developer productivity.",
            "DevSecOps Engineer integrating security into CI/CD pipelines using SAST, DAST, container scanning, and policy-as-code with OPA.",
        ],
        "projects": [
            "Migrated monolithic app to microservices on AWS EKS using Terraform IaC, ArgoCD GitOps, and Istio service mesh.",
            "Built multi-cloud disaster recovery system across AWS + Azure with automated failover achieving RTO < 5min.",
            "Implemented GitOps pipeline using ArgoCD + Flux reducing deployment frequency from weekly to 20x daily.",
            "Kubernetes cluster optimization reducing infrastructure costs by 35% using VPA, HPA, Karpenter, and spot instances.",
            "SRE dashboard with SLO tracking, error budget burn rate alerting, and automated runbook execution using Grafana + PagerDuty.",
            "Zero-trust network architecture using Cloudflare Access, HashiCorp Vault, and mTLS for all service-to-service communication.",
        ],
        "internships": [
            "DevOps Intern at Atlassian: Automated infrastructure provisioning reducing new environment setup from 2 days to 20 minutes.",
            "Cloud Engineering Intern at Deloitte: Migrated 15 microservices to Azure AKS with Helm charts and CI/CD pipelines.",
            "SRE Intern at Razorpay: Built alerting runbooks and automated 30% of incident response actions.",
        ],
    },

    "CYBERSECURITY": {
        "titles": [
            "Cybersecurity Engineer", "Security Analyst", "Penetration Tester",
            "Security Operations Center (SOC) Analyst", "Application Security Engineer",
            "Cloud Security Engineer", "Ethical Hacker", "Security Architect",
        ],
        "core_skills": [
            "network security", "penetration testing", "ethical hacking",
            "kali linux", "metasploit", "burp suite", "owasp top 10",
            "siem", "splunk", "incident response", "threat hunting",
            "vulnerability assessment", "nmap", "wireshark",
            "cryptography", "pki", "ssl/tls", "zero trust",
            "iam", "sso", "mfa", "oauth2", "active directory",
            "cloud security", "aws security", "azure security",
            "docker security", "kubernetes security", "devsecops",
            "python", "bash", "linux", "firewalls", "ids/ips",
        ],
        "optional_skills": [
            "maltego", "nessus", "openvas", "cobalt strike",
            "ghidra", "ida pro", "reverse engineering", "malware analysis",
            "honeypots", "threat intelligence", "mitre att&ck",
            "soar", "xdr", "endpoint detection", "crowdstrike",
            "compliance", "gdpr", "hipaa", "pci-dss", "iso 27001",
            "forensics", "osint", "social engineering",
        ],
        "certs": [
            "CompTIA Security+",
            "Certified Ethical Hacker (CEH)",
            "Offensive Security Certified Professional (OSCP)",
            "Certified Information Systems Security Professional (CISSP)",
            "AWS Certified Security Specialty",
            "Google Professional Cloud Security Engineer",
            "CompTIA CySA+",
        ],
        "summaries": [
            "Cybersecurity Engineer with expertise in penetration testing, cloud security, and DevSecOps. Certified ethical hacker with experience securing enterprise applications and infrastructure.",
            "SOC Analyst with 3+ years in threat hunting, incident response, and SIEM management using Splunk and Microsoft Sentinel.",
            "Application Security Engineer specializing in secure SDLC, OWASP vulnerability remediation, and automated security testing in CI/CD pipelines.",
            "Cloud Security Engineer with deep AWS and Azure security expertise — implementing zero-trust architecture, IAM governance, and compliance automation.",
            "Offensive security professional with OSCP certification, 5+ years conducting red team assessments for Fortune 500 companies.",
        ],
        "projects": [
            "Automated vulnerability scanner integrating Nessus + custom Python scripts with Jira ticketing for risk-prioritized remediation.",
            "Zero-trust network architecture implementation using HashiCorp Vault, service mesh mTLS, and OPA policy enforcement.",
            "Security monitoring platform using ELK Stack ingesting 100GB/day of logs with ML-based anomaly detection.",
            "CTF platform built for university cybersecurity training with 50+ challenges across web, crypto, forensics, and reverse engineering.",
            "DevSecOps pipeline with SAST (Semgrep), DAST (OWASP ZAP), SCA (Snyk), and container scanning (Trivy) integrated into GitHub Actions.",
            "Phishing simulation platform for employee security awareness training with reporting dashboard.",
        ],
        "internships": [
            "Security Intern at Infosys: Conducted web application penetration testing for 10 client applications finding 47 critical vulnerabilities.",
            "SOC Analyst Intern at Wipro: Monitored SIEM alerts, triaged incidents, and created 15 detection rules for Splunk.",
            "Cybersecurity R&D Intern at CDAC: Researched adversarial ML attacks on intrusion detection systems.",
        ],
    },

    "MOBILE-DEVELOPMENT": {
        "titles": [
            "Mobile App Developer", "React Native Developer",
            "Flutter Developer", "Android Developer", "iOS Developer",
            "Cross-Platform Mobile Developer", "Senior Mobile Engineer",
            "Mobile Architect",
        ],
        "core_skills": [
            "react native", "flutter", "dart", "javascript", "typescript",
            "android", "kotlin", "java", "ios", "swift", "swiftui",
            "firebase", "push notifications", "mobile ui",
            "rest api", "graphql", "state management", "redux",
            "bluetooth", "gps", "camera", "offline storage",
            "sqlite", "realm", "async storage", "secure storage",
            "app store optimization", "crashlytics", "analytics",
            "ci/cd", "fastlane", "git", "agile",
        ],
        "optional_skills": [
            "kotlin multiplatform", "ionic", "capacitor", "cordova",
            "expo", "detox", "appium", "maestro",
            "animations", "reanimated", "lottie", "skia",
            "webrtc", "in-app purchases", "stripe mobile sdk",
            "ar/vr", "arkit", "arcore", "unity",
            "wear os", "watchos", "tvos",
        ],
        "certs": [
            "Google Associate Android Developer",
            "Apple Developer Certification",
            "Meta React Native Developer Certificate",
            "Flutter & Dart Development Bootcamp Certificate",
            "AWS Mobile Development Certificate",
        ],
        "summaries": [
            "React Native Developer with 3+ years building cross-platform mobile apps for iOS and Android. Expert in complex animations, offline sync, and app store deployment.",
            "Flutter Developer specializing in high-performance, pixel-perfect mobile apps with custom UI components, animation, and Firebase integration.",
            "Senior Android Developer with Kotlin expertise building scalable apps with Jetpack Compose, MVVM architecture, and Room database.",
            "iOS Developer specializing in SwiftUI, Combine, and CoreData. Experience publishing 5+ apps on the App Store with 500k+ combined downloads.",
            "Cross-platform mobile architect with expertise in React Native and Flutter, designing shared business logic layers and platform-specific native modules.",
        ],
        "projects": [
            "Food delivery app in React Native with real-time GPS tracking, payment integration, and push notifications — 100k+ downloads.",
            "Fitness tracking app in Flutter with BLE integration for wearables, custom chart animations, and offline workout sync.",
            "E-commerce app in Kotlin (Jetpack Compose) with AR try-on feature using ARCore and 3D model rendering.",
            "Video calling app using React Native + WebRTC with background blur and screen sharing — supports 50 concurrent participants.",
            "Mental health journaling app in SwiftUI with Core ML emotion detection from text and HealthKit integration.",
        ],
        "internships": [
            "Mobile Dev Intern at Meesho: Built product image zoom and 360° view feature in React Native serving 50M users.",
            "Flutter Intern at Cure.fit: Developed workout progress screens with custom animations and Firebase offline sync.",
            "Android Intern at MakeMyTrip: Implemented seat selection UI for bus booking with real-time availability updates.",
        ],
    },

    "UI-UX": {
        "titles": [
            "UI/UX Designer", "Product Designer", "UX Researcher",
            "Interaction Designer", "Visual Designer", "UX Lead",
            "Design Systems Lead", "UI Engineer",
        ],
        "core_skills": [
            "figma", "adobe xd", "sketch", "prototyping", "wireframing",
            "user research", "usability testing", "information architecture",
            "interaction design", "visual design", "typography",
            "color theory", "design systems", "component libraries",
            "accessibility", "wcag", "responsive design", "mobile design",
            "css", "html", "after effects", "photoshop", "illustrator",
            "user journey mapping", "persona creation", "heuristic evaluation",
        ],
        "optional_skills": [
            "framer", "principle", "invision", "zeplin", "marvel",
            "miro", "notion", "airtable", "jira",
            "react", "javascript", "css-in-js", "storybook",
            "motion design", "micro-interactions", "haptic design",
            "ar/vr ux", "voice ui", "conversational design",
            "eye tracking", "a/b testing", "google analytics",
        ],
        "certs": [
            "Google UX Design Professional Certificate",
            "Interaction Design Foundation UX Certificate",
            "Adobe Certified Professional in UI Design",
            "Nielsen Norman Group UX Certification",
            "Figma Professional Certificate",
            "HFI Certified Usability Analyst (CUA)",
        ],
        "summaries": [
            "UI/UX Designer with 4+ years crafting user-centered digital products. Expert in Figma, design systems, and iterative user research to drive measurable UX improvements.",
            "Product Designer with a strong background in user research, interaction design, and visual design. Experienced in collaborating with engineering teams to ship polished products.",
            "Design Systems Lead with expertise in building and maintaining scalable component libraries in Figma with Storybook integration and accessibility compliance.",
            "UX Researcher specializing in mixed-methods research including user interviews, usability testing, and quantitative analysis to inform product strategy.",
            "UI Engineer who bridges design and development — creating pixel-perfect React components from Figma designs with micro-interaction animations and WCAG compliance.",
        ],
        "projects": [
            "Redesigned fintech app onboarding flow increasing activation rate by 35% through journey mapping and iterative A/B testing.",
            "Built comprehensive design system with 200+ Figma components and React Storybook with WCAG 2.1 AA compliance.",
            "UX audit of enterprise SaaS dashboard identifying 23 critical usability issues; redesign reduced task completion time by 28%.",
            "Gamified learning app UI/UX with motion design and micro-interactions achieving 4.8★ App Store rating.",
            "Conversational AI chatbot UX design with intent flow mapping, error state handling, and rapid prototype testing.",
            "Conducted 40+ user research sessions for a healthcare platform leading to complete navigation restructure improving findability by 60%.",
        ],
        "internships": [
            "UX Design Intern at Nykaa: Redesigned checkout flow reducing cart abandonment by 18% through usability testing insights.",
            "Product Design Intern at Groww: Created onboarding screens for SIP feature — shipped to 5M+ users after A/B testing.",
            "UI Intern at Hotstar: Designed dark mode for streaming app, contributing to 12% increase in night-time session duration.",
        ],
    },
}

# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def _make_raw_text(title: str, summary: str, skills: list[str],
                   projects: list[str], certs: list[str],
                   internships: list[str], role_key: str) -> str:
    """Build a realistic resume raw_text string."""
    universities = [
        "IIT Bombay", "IIT Delhi", "IIT Madras", "NIT Trichy", "NIT Surathkal",
        "BITS Pilani", "VIT Vellore", "SRM University", "Amrita University",
        "DTU Delhi", "IIIT Hyderabad", "IIT Kharagpur", "Manipal Institute of Technology",
        "PSG College of Technology", "Anna University",
    ]
    degrees = [
        "B.Tech in Computer Science", "B.E. in Computer Science",
        "B.Tech in Information Technology", "M.Tech in Computer Science",
        "M.S. in Artificial Intelligence", "M.S. in Data Science",
        "B.Tech in Electronics and Communication",
        "MCA (Master of Computer Applications)",
    ]
    companies = [
        "TCS", "Infosys", "Wipro", "HCL", "Cognizant", "Accenture",
        "IBM", "Oracle", "SAP", "Capgemini", "Tech Mahindra",
        "Amazon", "Google India", "Microsoft India", "Samsung R&D",
        "Flipkart", "Zomato", "Swiggy", "BYJU'S", "Razorpay",
        "PhonePe", "Paytm", "MakeMyTrip", "Ola", "Freshworks",
    ]
    yoe = random.randint(1, 8)
    company = random.choice(companies)
    uni = random.choice(universities)
    degree = random.choice(degrees)
    grad_year = random.randint(2016, 2023)

    skills_str = ", ".join(random.sample(skills, min(len(skills), random.randint(8, 14))))
    projects_str = "; ".join(random.sample(projects, min(len(projects), random.randint(2, 3))))
    certs_str = "\n".join(random.sample(certs, min(len(certs), random.randint(1, 3))))
    intern_str = random.choice(internships) if internships else ""

    text = f"""{title}
Summary
{summary}

Highlights
{skills_str}

Experience
{title} | {company} | {grad_year + 1} – Present
- Developed and deployed production-grade solutions impacting {random.randint(10,500)}k+ users.
- Collaborated with cross-functional teams in Agile sprints using Jira and Confluence.
- Mentored junior developers and contributed to code reviews and architectural decisions.
- Led technical design discussions and contributed to system design documents.
- Integrated third-party APIs and cloud services improving system reliability.

{"Junior " + title} | {random.choice(companies)} | {grad_year} – {grad_year + 1}
- Built features and fixed bugs in production systems used by enterprise clients.
- Wrote unit tests achieving 85%+ code coverage using pytest/jest.
- Participated in on-call rotation and incident response.

Projects
{projects_str}

Education
{degree} | {uni} | {grad_year}
CGPA: {round(random.uniform(7.2, 9.5), 2)}

Certifications
{certs_str}

Internships
{intern_str}
"""
    return text.strip()


def generate_resumes_for_role(role_key: str, n: int = 65) -> list[dict]:
    data = ROLE_DATA[role_key]
    rows = []
    for i in range(n):
        title = random.choice(data["titles"])
        summary = random.choice(data["summaries"])
        all_skills = data["core_skills"] + data["optional_skills"]

        # pick skills for this resume
        n_skills = random.randint(8, 18)
        # always include most core skills
        skill_pool = random.sample(data["core_skills"], min(len(data["core_skills"]), random.randint(6, len(data["core_skills"]))))
        skill_pool += random.sample(data["optional_skills"], min(len(data["optional_skills"]), random.randint(2, 6)))
        selected_skills = list(set(skill_pool))[:n_skills]

        project = random.choice(data["projects"])
        cert = random.choice(data["certs"]) if random.random() > 0.2 else None
        intern = random.choice(data["internships"]) if random.random() > 0.1 else None

        raw = _make_raw_text(
            title=title,
            summary=summary,
            skills=selected_skills,
            projects=data["projects"],
            certs=data["certs"],
            internships=data["internships"],
            role_key=role_key,
        )

        rows.append({
            "resume_id": f"cse_{role_key.lower()}_{i+1:04d}.pdf",
            "domain": role_key,
            "raw_text": raw,
            "skills": ", ".join(selected_skills),
            "projects": project,
            "certifications": cert if cert else "Certifications",
            "internships": intern,
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    csv_path = os.path.join(os.path.dirname(__file__), "data", "structured_resumes.csv")

    print(f"Loading existing dataset: {csv_path}")
    df_existing = pd.read_csv(csv_path)
    print(f"  Existing rows: {len(df_existing)}")

    # Remove any previously generated CSE rows (re-run safe)
    cse_domains = set(ROLE_DATA.keys())
    df_existing = df_existing[~df_existing["domain"].isin(cse_domains)]
    print(f"  Rows after removing old CSE rows: {len(df_existing)}")

    # Generate new rows
    all_new = []
    for role_key in ROLE_DATA:
        rows = generate_resumes_for_role(role_key, n=70)
        all_new.extend(rows)
        print(f"  Generated {len(rows)} resumes for {role_key}")

    df_new = pd.DataFrame(all_new)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_csv(csv_path, index=False)

    print(f"\nDone! Updated dataset saved.")
    print(f"  Total rows: {len(df_combined)}")
    print(f"  New CSE rows added: {len(df_new)}")
    print(f"\nDomain distribution (CSE + existing):")
    print(df_combined["domain"].value_counts().to_string())


if __name__ == "__main__":
    main()
