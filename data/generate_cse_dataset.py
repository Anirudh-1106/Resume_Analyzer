"""
generate_cse_dataset.py
Generates a clean synthetic CSE resume dataset with genuine job_role labels.
Replaces structured_resumes.csv with 3,200 properly labelled resumes.

Run from data/ folder:
    python generate_cse_dataset.py
"""

import os
import uuid
import random
import pandas as pd

random.seed(42)

# ---------------------------------------------------------------------------
# Role definitions — required/optional skills, certs, summaries, projects
# ---------------------------------------------------------------------------
ROLES = {
    "Data Scientist": {
        "required": [
            "python", "machine learning", "deep learning", "pandas", "numpy",
            "scikit-learn", "sql", "statistics", "data analysis", "matplotlib",
        ],
        "optional": [
            "tensorflow", "pytorch", "nlp", "tableau", "r", "spark",
            "xgboost", "lightgbm", "seaborn", "plotly", "mlflow",
            "feature engineering", "hypothesis testing", "pca", "clustering",
        ],
        "certs": [
            "Google Data Analytics Certificate",
            "IBM Data Science Professional Certificate",
            "Coursera Machine Learning Specialization",
            "DataCamp Data Scientist Professional",
            "AWS Certified Machine Learning Specialty",
            "Microsoft Azure Data Scientist Associate",
        ],
        "summaries": [
            "Results-driven Data Scientist with expertise in predictive modeling and data analysis using Python and ML frameworks.",
            "Data Scientist skilled in statistical modeling, A/B testing, and building end-to-end ML pipelines.",
            "Analytical Data Scientist experienced in NLP, time-series forecasting, and interactive dashboards.",
        ],
        "projects": [
            "Customer Churn Prediction with XGBoost achieving 93% accuracy",
            "Sales Forecasting Model using LSTM reducing forecast error by 38%",
            "Sentiment Analysis pipeline using BERT fine-tuned on 500k reviews",
            "Fraud Detection System processing 10M transactions per day",
            "Recommendation Engine using collaborative filtering for e-commerce",
            "COVID-19 Spread Prediction dashboard using ML epidemiological models",
        ],
        "internships": [
            "Data Science Intern at Google — built A/B testing framework",
            "ML Research Intern at Microsoft — contributed to NLP summarization",
            "Data Science Intern at Flipkart — built product recommendation engine",
            "Analytics Intern at Accenture — automated reporting pipeline",
        ],
    },

    "Machine Learning Engineer": {
        "required": [
            "python", "machine learning", "deep learning", "tensorflow", "pytorch",
            "scikit-learn", "docker", "git", "mlflow", "sql",
        ],
        "optional": [
            "kubeflow", "kubernetes", "aws", "spark", "kafka", "onnx",
            "triton", "fastapi", "airflow", "redis", "xgboost",
            "feature store", "bentoml", "seldon", "ci/cd",
        ],
        "certs": [
            "AWS Certified Machine Learning Specialty",
            "TensorFlow Developer Certificate",
            "Deep Learning Specialization — Coursera",
            "MLOps Specialization — Coursera",
            "Google Professional Machine Learning Engineer",
        ],
        "summaries": [
            "MLOps-focused ML Engineer building and monitoring production ML systems at scale.",
            "ML Engineer specializing in model optimization, feature stores, and real-time inference APIs.",
            "Applied ML Engineer with experience in distributed training and automated retraining pipelines.",
        ],
        "projects": [
            "End-to-end MLOps pipeline on AWS SageMaker with automated retraining on data drift",
            "Real-time model serving on Kubernetes with Triton handling 50k req/sec",
            "Feature store using Feast integrated with Kafka and Redis",
            "GPU-optimized LLM fine-tuning pipeline using DeepSpeed ZeRO",
            "AutoML hyperparameter search system using Optuna across 50+ model configs",
            "Canary deployment framework for ML models reducing rollback time from hours to minutes",
        ],
        "internships": [
            "ML Engineering Intern at Amazon — automated model evaluation harness",
            "MLOps Intern at Uber — contributed to internal feature store",
            "ML Platform Intern at LinkedIn — built model monitoring dashboard",
        ],
    },

    "AI Engineer": {
        "required": [
            "python", "deep learning", "nlp", "transformers", "pytorch",
            "langchain", "openai api", "hugging face", "llm", "generative ai",
        ],
        "optional": [
            "llamaindex", "rag", "stable diffusion", "lora", "rlhf",
            "vector databases", "fastapi", "docker", "aws", "onnx",
            "prompt engineering", "semantic search", "embeddings", "fine-tuning",
        ],
        "certs": [
            "DeepLearning.AI LLM Specialization",
            "Hugging Face NLP Course Certificate",
            "AWS Certified AI Practitioner",
            "Google Cloud Professional ML Engineer",
            "Microsoft Azure AI Engineer Associate",
        ],
        "summaries": [
            "Generative AI Engineer building LLM fine-tuning, RAG pipelines, and AI agent systems.",
            "AI Engineer with expertise in LangChain, OpenAI API, and vector database integration.",
            "LLM Engineer focused on enterprise AI assistants, RAG systems, and prompt engineering.",
        ],
        "projects": [
            "Enterprise RAG chatbot using LangChain + Pinecone + GPT-4 over 10,000-page knowledge base",
            "Fine-tuned LLaMA-3 on legal dataset using QLoRA — 40% improvement on benchmarks",
            "Multi-agent AI research assistant using AutoGen with 5 specialized agents",
            "Text-to-image system using Stable Diffusion with ControlNet for e-commerce",
            "Semantic search over 1M product catalog using bi-encoder embeddings and FAISS",
            "AI code review assistant powered by GPT-4 integrated into GitHub PR workflow",
        ],
        "internships": [
            "AI Research Intern at OpenAI — contributed to RLHF data pipeline",
            "NLP Intern at Samsung Research — built intent classification for voice assistant",
            "Computer Vision Intern at Tesla — developed object tracking for autonomous driving",
            "AI Engineering Intern at Wipro — document extraction system using LayoutLM",
        ],
    },

    "Data Engineer": {
        "required": [
            "python", "sql", "spark", "kafka", "airflow", "docker",
            "postgresql", "aws", "git", "linux",
        ],
        "optional": [
            "dbt", "snowflake", "databricks", "delta lake", "kubernetes",
            "terraform", "redis", "elasticsearch", "mongodb", "scala",
            "bigquery", "redshift", "flink", "hive",
        ],
        "certs": [
            "Google Professional Data Engineer",
            "AWS Certified Data Analytics Specialty",
            "Databricks Certified Associate Developer for Apache Spark",
            "dbt Analytics Engineering Certification",
            "Snowflake SnowPro Core Certification",
        ],
        "summaries": [
            "Data Engineer building scalable data pipelines and lakehouse architectures using Spark and cloud platforms.",
            "Senior Data Engineer specializing in real-time streaming systems with Kafka, Flink, and Delta Lake.",
            "Analytics Engineer building dbt models, semantic layers, and self-serve analytics platforms.",
        ],
        "projects": [
            "Real-time data lakehouse on Azure using Kafka + Databricks + Delta Lake processing 5TB/day",
            "dbt transformation layer over Snowflake with 300+ models and CI/CD via GitHub Actions",
            "Event-driven pipeline using Kafka Streams for financial transactions with exactly-once semantics",
            "Multi-cloud data migration from Oracle to AWS Redshift with zero downtime",
            "Data quality monitoring framework using Great Expectations integrated into Airflow DAGs",
            "IoT sensor streaming analytics using Apache Flink processing from 10k devices in real-time",
        ],
        "internships": [
            "Data Engineering Intern at Swiggy — ETL pipeline ingesting 50M daily order events into BigQuery",
            "Data Platform Intern at Razorpay — dbt models reducing reconciliation time by 60%",
            "Big Data Intern at TCS — Spark jobs optimization reducing runtime from 4hrs to 40min",
        ],
    },

    "Software Engineer": {
        "required": [
            "python", "java", "git", "sql", "rest api", "docker",
            "linux", "data structures", "algorithms", "postgresql",
        ],
        "optional": [
            "kubernetes", "aws", "redis", "mongodb", "grpc",
            "spring boot", "fastapi", "django", "microservices", "kafka",
            "c++", "go", "ci/cd", "unit testing",
        ],
        "certs": [
            "AWS Certified Solutions Architect Associate",
            "Oracle Java SE Programmer Certification",
            "Google Associate Cloud Engineer",
            "Kubernetes CKAD",
            "GitHub Actions Certification",
        ],
        "summaries": [
            "Software Engineer with expertise in microservices, APIs, and cloud deployments.",
            "Senior Software Engineer building scalable backend systems using Python, Java, and PostgreSQL.",
            "Full-cycle Software Engineer from architecture design to CI/CD and production monitoring.",
        ],
        "projects": [
            "Microservices e-commerce platform handling 1M daily orders",
            "REST API backend with rate limiting, JWT auth, and OpenTelemetry tracing",
            "Real-time chat application using WebSocket and Redis pub/sub",
            "Task management platform with role-based access control and audit trails",
            "Payment gateway integration with idempotency and retry logic",
            "Distributed job scheduler processing 500k background tasks per day",
        ],
        "internships": [
            "Software Engineering Intern at Infosys — built internal HR portal",
            "Backend Intern at Freshworks — bulk email campaign feature with SendGrid",
            "Engineering Intern at Zoho — REST API improvements for CRM product",
        ],
    },

    "Frontend Developer": {
        "required": [
            "javascript", "typescript", "react", "html", "css",
            "next.js", "git", "rest api", "tailwind", "webpack",
        ],
        "optional": [
            "vue.js", "angular", "graphql", "vite", "jest",
            "figma", "redux", "framer", "storybook", "websockets",
            "performance optimization", "accessibility", "pwa",
        ],
        "certs": [
            "Meta Front-End Developer Professional Certificate",
            "freeCodeCamp Responsive Web Design",
            "Google UX Design Certificate",
            "JavaScript Algorithms and Data Structures — freeCodeCamp",
            "Frontend Masters Performance Certificate",
        ],
        "summaries": [
            "Frontend Developer building performant, accessible React applications using TypeScript and Next.js.",
            "Senior Frontend Engineer specializing in large-scale React SPAs and micro-frontend architectures.",
            "Creative Frontend Developer with expertise in animations, WebGL, and interactive data visualizations.",
        ],
        "projects": [
            "E-commerce platform in Next.js 14 with Core Web Vitals score >95",
            "Design system with 80+ React components and Storybook documentation",
            "Real-time collaborative whiteboard using React, WebSocket, and Canvas API",
            "Dashboard with D3.js visualizing 1M+ data points with WebWorker aggregations",
            "PWA with offline support, push notifications — 100 Lighthouse score",
            "Micro-frontend architecture using Module Federation across 5 React apps",
        ],
        "internships": [
            "Frontend Intern at Razorpay — payment UI components reducing checkout abandonment by 12%",
            "React Developer Intern at OYO — hotel listing page improving LCP by 40%",
            "Web Development Intern at Internshala — interactive dashboard for 1M+ users",
        ],
    },

    "Full Stack Developer": {
        "required": [
            "javascript", "typescript", "react", "node.js", "python",
            "sql", "mongodb", "docker", "git", "rest api",
        ],
        "optional": [
            "next.js", "graphql", "aws", "redis", "kubernetes",
            "fastapi", "postgresql", "tailwind", "websockets", "kafka",
            "microservices", "authentication", "jwt",
        ],
        "certs": [
            "AWS Certified Developer Associate",
            "MongoDB Certified Developer",
            "Meta Full Stack Developer Professional Certificate",
            "freeCodeCamp Full Stack Development Certificate",
            "Udacity Full Stack Web Developer Nanodegree",
        ],
        "summaries": [
            "Full Stack Developer with 4+ years building end-to-end web applications using React and Node.js.",
            "MERN Stack Engineer skilled in scalable APIs, real-time features, and cloud deployment.",
            "Full Stack Engineer leading teams to deliver complex SaaS products using modern JS/TS stack.",
        ],
        "projects": [
            "SaaS project management tool with Next.js, FastAPI, PostgreSQL, and real-time WebSocket",
            "E-learning platform with video streaming, quiz engine, and Stripe payment integration",
            "Real-time food delivery tracking app with Google Maps API and React Native mobile frontend",
            "Multi-tenant CRM system with React, Django REST, PostgreSQL, and Redis caching",
            "Open-source note-taking app (Notion alternative) using Next.js and Supabase",
            "Social media platform with infinite scroll, media upload, and notification system",
        ],
        "internships": [
            "Full Stack Intern at Zoho — HR portal using React + Java Spring Boot for 5,000 employees",
            "Software Engineering Intern at Freshworks — bulk email campaign feature",
            "Full Stack Intern at startup — built MVP HealthTech app using MERN stack in 6 weeks",
        ],
    },

    "Backend Developer": {
        "required": [
            "python", "java", "sql", "rest api", "docker", "git",
            "postgresql", "redis", "linux", "microservices",
        ],
        "optional": [
            "go", "rust", "kafka", "grpc", "kubernetes", "aws",
            "mongodb", "elasticsearch", "spring boot", "fastapi",
            "message queues", "load balancing", "caching",
        ],
        "certs": [
            "AWS Certified Developer Associate",
            "Oracle Java SE Programmer Certification",
            "Google Professional Cloud Developer",
            "Docker Certified Associate",
            "Kubernetes CKAD",
        ],
        "summaries": [
            "Backend Engineer designing high-performance REST and gRPC APIs using Python, Go, and PostgreSQL.",
            "Senior Backend Developer specializing in distributed systems and database optimization.",
            "Python Backend Engineer building scalable APIs with FastAPI/Django serving millions of requests.",
        ],
        "projects": [
            "Distributed payments API in Go + PostgreSQL with idempotency processing 100k TPS",
            "Event-driven microservices platform using Java Spring Boot and Kafka with 12 services",
            "GraphQL federation gateway in Node.js unifying 8 downstream microservices",
            "Real-time notification service using WebSockets + Redis Pub/Sub for 500k concurrent connections",
            "High-performance search API using Elasticsearch + FastAPI with custom relevance tuning",
            "REST API gateway with rate limiting, circuit breakers, and distributed tracing",
        ],
        "internships": [
            "Backend Intern at CRED — rewards calculation engine handling 2M events/day",
            "API Development Intern at Paytm — UPI transaction status polling service",
            "Backend Engineering Intern at Juspay — idempotency layer for payment APIs",
        ],
    },

    "DevOps Engineer": {
        "required": [
            "docker", "kubernetes", "aws", "terraform", "linux",
            "git", "ci/cd", "python", "bash", "ansible",
        ],
        "optional": [
            "azure", "gcp", "helm", "argocd", "prometheus",
            "grafana", "jenkins", "github actions", "vault", "istio",
            "monitoring", "logging", "alerting", "gitops",
        ],
        "certs": [
            "AWS Certified DevOps Engineer Professional",
            "Certified Kubernetes Administrator (CKA)",
            "HashiCorp Certified Terraform Associate",
            "Docker Certified Associate",
            "Google Professional DevOps Engineer",
        ],
        "summaries": [
            "DevOps Engineer building robust CI/CD pipelines and cloud infrastructure for high-availability systems.",
            "Site Reliability Engineer maintaining 99.99% uptime through automated incident response and chaos engineering.",
            "Platform Engineer building internal developer platforms using Backstage, Kubernetes, and GitOps.",
        ],
        "projects": [
            "Monolith-to-microservices migration on AWS EKS using Terraform IaC and ArgoCD GitOps",
            "Multi-cloud disaster recovery across AWS + Azure with automated failover RTO < 5min",
            "GitOps pipeline using ArgoCD + Flux enabling 20x daily deployments",
            "Kubernetes cost optimization reducing infrastructure spend by 35% using Karpenter and spot instances",
            "SRE dashboard with SLO tracking and automated runbook execution via Grafana + PagerDuty",
            "Zero-trust network using Cloudflare Access, HashiCorp Vault, and mTLS for all services",
        ],
        "internships": [
            "DevOps Intern at Atlassian — infrastructure provisioning reducing setup from 2 days to 20 min",
            "Cloud Engineering Intern at Deloitte — migrated 15 microservices to Azure AKS",
            "SRE Intern at Razorpay — automated 30% of incident response actions",
        ],
    },

    "Cloud Engineer": {
        "required": [
            "aws", "azure", "terraform", "docker", "kubernetes",
            "linux", "python", "git", "networking", "security",
        ],
        "optional": [
            "gcp", "ansible", "helm", "prometheus", "grafana",
            "cloudformation", "lambda", "s3", "vpc", "iam",
            "cost optimization", "finops", "serverless",
        ],
        "certs": [
            "AWS Solutions Architect Professional",
            "Microsoft Azure Administrator Associate",
            "Google Cloud Professional Cloud Architect",
            "AWS Certified DevOps Engineer",
            "HashiCorp Certified Terraform Associate",
        ],
        "summaries": [
            "Cloud Architect designing multi-region, fault-tolerant architectures on AWS and Azure.",
            "Cloud Engineer migrating on-premise infrastructure to cloud platforms with zero downtime.",
            "Cloud Solutions Engineer optimizing cloud costs and improving infrastructure reliability.",
        ],
        "projects": [
            "Multi-region cloud architecture with auto-scaling and 99.99% SLA on AWS",
            "Serverless application using AWS Lambda, API Gateway, and DynamoDB handling 1M req/day",
            "Cloud cost optimization reducing monthly AWS spend by $50k using Reserved Instances and rightsizing",
            "Disaster recovery system across AWS + GCP with automated failover",
            "Kubernetes multi-cluster management using Rancher across 3 cloud providers",
            "Cloud security hardening achieving SOC 2 Type II compliance",
        ],
        "internships": [
            "Cloud Intern at Capgemini — Azure migration for enterprise client saving 40% on infra costs",
            "AWS Intern at Wipro — serverless ETL pipeline using Lambda and Glue",
            "Cloud Engineering Intern at Accenture — IaC templates for 30+ AWS services",
        ],
    },

    "Cybersecurity Engineer": {
        "required": [
            "python", "linux", "networking", "penetration testing",
            "ethical hacking", "owasp", "siem", "git", "bash", "sql",
        ],
        "optional": [
            "metasploit", "wireshark", "oscp", "zero trust",
            "devsecops", "docker", "aws", "threat modeling",
            "cryptography", "incident response", "kali linux",
        ],
        "certs": [
            "CompTIA Security+",
            "Certified Ethical Hacker (CEH)",
            "Offensive Security Certified Professional (OSCP)",
            "CISSP",
            "AWS Certified Security Specialty",
        ],
        "summaries": [
            "Cybersecurity Engineer specializing in penetration testing, cloud security, and DevSecOps.",
            "SOC Analyst with expertise in threat hunting, incident response, and SIEM management.",
            "Application Security Engineer integrating security into SDLC and CI/CD pipelines.",
        ],
        "projects": [
            "Automated vulnerability scanner integrating Nessus with Jira for risk-prioritized remediation",
            "Zero-trust network architecture using HashiCorp Vault and OPA policy enforcement",
            "Security monitoring platform using ELK Stack with ML-based anomaly detection on 100GB/day logs",
            "CTF platform with 50+ challenges for university cybersecurity training",
            "DevSecOps pipeline with SAST, DAST, SCA, and container scanning in GitHub Actions",
            "Phishing simulation platform for employee security awareness training",
        ],
        "internships": [
            "Security Intern at Infosys — web application penetration testing finding 47 critical vulnerabilities",
            "SOC Analyst Intern at Wipro — SIEM monitoring and 15 detection rules for Splunk",
            "Cybersecurity R&D Intern at CDAC — adversarial ML attacks on intrusion detection systems",
        ],
    },

    "Mobile App Developer": {
        "required": [
            "flutter", "dart", "react native", "javascript",
            "firebase", "git", "rest api", "android", "ios", "sql",
        ],
        "optional": [
            "swift", "kotlin", "swiftui", "jetpack compose",
            "redux", "graphql", "aws", "push notifications",
            "offline sync", "animations", "biometrics",
        ],
        "certs": [
            "Google Associate Android Developer",
            "Apple iOS Developer Certification",
            "Flutter Development Bootcamp Certificate",
            "React Native Specialization — Coursera",
            "Firebase Developer Certification",
        ],
        "summaries": [
            "React Native Developer with expertise in cross-platform mobile apps with complex animations.",
            "Flutter Developer building pixel-perfect mobile apps with custom UI and Firebase integration.",
            "Senior Android Developer with Kotlin expertise using Jetpack Compose and MVVM architecture.",
        ],
        "projects": [
            "Food delivery app in React Native with real-time GPS tracking and payment integration — 100k+ downloads",
            "Fitness tracking app in Flutter with BLE wearable integration and offline workout sync",
            "E-commerce app in Kotlin with ARCore try-on feature and 3D model rendering",
            "Video calling app using React Native + WebRTC with background blur supporting 50 participants",
            "Mental health journaling app in SwiftUI with Core ML emotion detection",
            "Travel booking app with offline map support and real-time availability updates",
        ],
        "internships": [
            "Mobile Dev Intern at Meesho — product image zoom feature in React Native serving 50M users",
            "Flutter Intern at Cure.fit — workout progress screens with custom animations",
            "Android Intern at MakeMyTrip — seat selection UI with real-time bus availability",
        ],
    },

    "UI/UX Designer": {
        "required": [
            "figma", "adobe xd", "wireframing", "prototyping",
            "user research", "design systems", "html", "css",
            "usability testing", "typography",
        ],
        "optional": [
            "sketch", "framer", "invision", "zeplin", "javascript",
            "react", "motion design", "accessibility", "design thinking",
            "information architecture", "color theory",
        ],
        "certs": [
            "Google UX Design Professional Certificate",
            "Interaction Design Foundation UX Certificate",
            "Adobe Certified Professional in UI Design",
            "Nielsen Norman Group UX Certificate",
            "Figma Professional Certificate",
        ],
        "summaries": [
            "UI/UX Designer crafting user-centered digital products with Figma and iterative user research.",
            "Product Designer with strong background in user research and visual design for mobile and web.",
            "Design Systems Lead building scalable component libraries with Storybook and accessibility compliance.",
        ],
        "projects": [
            "Fintech app onboarding redesign increasing activation rate by 35% through A/B testing",
            "Design system with 200+ Figma components and WCAG 2.1 AA compliance",
            "Enterprise SaaS dashboard UX audit reducing task completion time by 28%",
            "Gamified learning app UI with motion design achieving 4.8★ App Store rating",
            "Conversational AI chatbot UX with intent flow mapping and error state handling",
            "Healthcare platform navigation redesign improving findability by 60% after 40+ user interviews",
        ],
        "internships": [
            "UX Design Intern at Nykaa — checkout flow redesign reducing cart abandonment by 18%",
            "Product Design Intern at Groww — onboarding screens for SIP feature shipped to 5M+ users",
            "UI Intern at Hotstar — dark mode design contributing to 12% increase in night session duration",
        ],
    },

    "Business Analyst": {
        "required": [
            "sql", "excel", "tableau", "power bi", "python",
            "data analysis", "requirements gathering", "agile",
            "stakeholder management", "documentation",
        ],
        "optional": [
            "r", "jira", "confluence", "salesforce", "sap",
            "machine learning", "forecasting", "ab testing",
            "looker", "google analytics", "process mapping",
        ],
        "certs": [
            "CBAP Certification",
            "PMI Professional in Business Analysis (PMI-PBA)",
            "Tableau Desktop Specialist",
            "Power BI Data Analyst Associate",
            "Agile Business Analysis Certificate",
        ],
        "summaries": [
            "Business Analyst bridging technical teams and business stakeholders to drive data-driven decisions.",
            "Senior BA with expertise in process optimization, KPI dashboards, and requirements elicitation.",
            "Data-driven Business Analyst experienced in Tableau, Power BI, and statistical analysis.",
        ],
        "projects": [
            "Business process optimization reducing operational costs by 22% through workflow automation",
            "KPI dashboard in Tableau tracking 30+ metrics across 5 business units",
            "Market segmentation analysis using clustering identifying 4 new customer segments",
            "Requirements documentation for ERP migration serving 2,000+ employees",
            "ROI analysis model for digital marketing spend optimizing $2M annual budget",
            "Churn analysis report identifying at-risk customers and improving retention by 15%",
        ],
        "internships": [
            "Business Analyst Intern at Deloitte — process mapping saving 15 hours/week in reporting",
            "BA Intern at KPMG — stakeholder requirements for banking digital transformation",
            "Data Analyst Intern at Nielsen — consumer insights dashboard for FMCG client",
        ],
    },

    "NLP Engineer": {
        "required": [
            "python", "nlp", "transformers", "pytorch", "tensorflow",
            "hugging face", "spacy", "nltk", "deep learning", "sql",
        ],
        "optional": [
            "langchain", "openai api", "llm", "bert", "gpt",
            "text classification", "named entity recognition",
            "machine translation", "docker", "aws",
            "fine-tuning", "embeddings", "semantic search",
        ],
        "certs": [
            "DeepLearning.AI NLP Specialization",
            "Hugging Face NLP Course Certificate",
            "Stanford NLP Certificate",
            "Coursera Natural Language Processing Specialization",
            "Google Cloud NLP API Developer",
        ],
        "summaries": [
            "NLP Engineer specializing in transformer models, text classification, and production NLP systems.",
            "Senior NLP Engineer with expertise in BERT fine-tuning, information extraction, and semantic search.",
            "Applied NLP researcher building conversational AI systems and document understanding pipelines.",
        ],
        "projects": [
            "Sentiment Analysis Engine classifying 1M+ customer reviews with 94% F1 score",
            "Named Entity Recognition system for medical records using BioBERT",
            "Machine Translation model English-to-Hindi using MarianMT achieving BLEU 42",
            "Question Answering system over enterprise documents using extractive QA + RAG",
            "Text Summarization pipeline using T5 reducing support ticket response time by 50%",
            "Multilingual chatbot supporting 8 languages using mBERT and RASA framework",
        ],
        "internships": [
            "NLP Intern at Amazon Alexa — intent classification improving accuracy from 88% to 95%",
            "Research Intern at IIT Madras NLP Lab — cross-lingual document clustering",
            "NLP Engineering Intern at Vernacular.ai — ASR post-processing pipeline",
        ],
    },

    "Computer Vision Engineer": {
        "required": [
            "python", "computer vision", "deep learning", "pytorch",
            "tensorflow", "opencv", "numpy", "scikit-learn", "git", "docker",
        ],
        "optional": [
            "yolo", "resnet", "image segmentation", "object detection",
            "aws", "cuda", "onnx", "mlflow", "generative ai",
            "3d vision", "point cloud", "lidar",
        ],
        "certs": [
            "DeepLearning.AI Computer Vision Specialization",
            "OpenCV University Certification",
            "NVIDIA Deep Learning Institute Certificate",
            "AWS Certified Machine Learning Specialty",
            "Coursera Computer Vision Specialization",
        ],
        "summaries": [
            "Computer Vision Engineer with expertise in object detection, segmentation, and real-time video analytics.",
            "CV Engineer deploying production vision models using YOLO, ResNet, and Detectron2.",
            "Applied Computer Vision researcher working on medical imaging, autonomous systems, and generative models.",
        ],
        "projects": [
            "Real-time object detection system using YOLOv8 with 97% precision for manufacturing defects",
            "Face recognition authentication system using ArcFace achieving FAR < 0.01%",
            "Medical image segmentation for tumor detection using U-Net on MRI scans",
            "Autonomous vehicle pedestrian detection pipeline achieving 40ms latency on edge hardware",
            "GAN-based synthetic data generator augmenting training datasets by 5x",
            "Real-time pose estimation system using MediaPipe for fitness coaching app",
        ],
        "internships": [
            "CV Intern at Qualcomm — on-device inference optimization reducing model size by 60%",
            "Research Intern at IISc — stereo depth estimation for indoor robotics navigation",
            "Computer Vision Intern at Jio — crowd density estimation for smart city surveillance",
        ],
    },
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
UNIVERSITIES = [
    "IIT Bombay", "IIT Delhi", "IIT Madras", "IIT Kharagpur", "IIT Roorkee",
    "NIT Trichy", "NIT Surathkal", "NIT Warangal", "BITS Pilani",
    "VIT Vellore", "SRM University", "Amrita University",
    "DTU Delhi", "IIIT Hyderabad", "Manipal Institute of Technology",
    "PSG College of Technology", "Anna University", "COEP Pune",
]
DEGREES = [
    "B.Tech in Computer Science Engineering",
    "B.E. in Computer Science",
    "B.Tech in Information Technology",
    "M.Tech in Computer Science",
    "M.S. in Artificial Intelligence",
    "M.S. in Data Science",
    "B.Tech in Electronics and Communication Engineering",
    "MCA (Master of Computer Applications)",
]
COMPANIES = [
    "TCS", "Infosys", "Wipro", "HCL Technologies", "Cognizant", "Accenture",
    "IBM India", "Oracle India", "Capgemini", "Tech Mahindra",
    "Amazon India", "Google India", "Microsoft India", "Samsung R&D India",
    "Flipkart", "Zomato", "Swiggy", "BYJU'S", "Razorpay", "PhonePe",
    "Paytm", "MakeMyTrip", "Ola", "Freshworks", "CRED", "Juspay",
]


def _make_raw_text(role: str, data: dict, skills: list, projects: list,
                   certs: list, internship: str, grad_year: int) -> str:
    summary = random.choice(data["summaries"])
    company = random.choice(COMPANIES)
    prev_company = random.choice(COMPANIES)
    uni = random.choice(UNIVERSITIES)
    degree = random.choice(DEGREES)
    yoe = random.randint(1, 7)

    proj_text = "\n".join(f"  - {p}" for p in projects)
    cert_text = "\n".join(f"  - {c}" for c in certs) if certs else "  - None"
    intern_text = f"  - {internship}" if internship else "  - No formal internship"

    return f"""{role}

Summary
{summary}

Technical Skills
{', '.join(skills)}

Professional Experience
{role} | {company} | {grad_year + 1} – Present ({yoe} yrs)
- Designed and deployed production-grade systems impacting {random.randint(10, 500)}k+ users.
- Collaborated with cross-functional teams across Agile sprints using Jira and Confluence.
- Led code reviews, system design sessions, and mentored 2–4 junior engineers.
- Integrated cloud services improving system uptime to 99.9%.

Junior {role} | {prev_company} | {grad_year} – {grad_year + 1}
- Built and maintained features for enterprise clients in production environments.
- Achieved 85%+ test coverage using automated testing frameworks.
- Participated in on-call rotation and post-mortem documentation.

Projects
{proj_text}

Education
{degree} | {uni} | {grad_year}
CGPA: {round(random.uniform(7.2, 9.6), 2)} / 10.0

Certifications
{cert_text}

Internships / Training
{intern_text}
""".strip()


def generate_resume(role: str, data: dict) -> dict:
    required = data["required"]
    optional = data["optional"]

    # Skills: all required (shuffled) + random subset of optional
    n_optional = random.randint(3, min(8, len(optional)))
    skills = list(set(required + random.sample(optional, n_optional)))
    random.shuffle(skills)

    # Projects: 2–4 per resume
    n_projects = random.randint(2, min(4, len(data["projects"])))
    projects = random.sample(data["projects"], n_projects)

    # Certs: 0–3
    n_certs = random.randint(0, min(3, len(data["certs"])))
    certs = random.sample(data["certs"], n_certs)

    # Internship: 75% chance of having one
    internship = random.choice(data["internships"]) if random.random() > 0.25 else ""

    grad_year = random.randint(2018, 2024)

    raw_text = _make_raw_text(role, data, skills, projects, certs, internship, grad_year)

    domain = role.upper().replace(" ", "-").replace("/", "-")

    return {
        "resume_id":       str(uuid.uuid4())[:12],
        "domain":          domain,
        "job_role":        role,
        "raw_text":        raw_text,
        "skills":          ", ".join(skills),
        "projects":        "; ".join(projects),
        "certifications":  "; ".join(certs) if certs else "",
        "internships":     internship,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
SAMPLES_PER_ROLE = 200   # 200 × 16 roles = 3,200 resumes

if __name__ == "__main__":
    import uuid

    records = []
    for role, data in ROLES.items():
        for _ in range(SAMPLES_PER_ROLE):
            records.append(generate_resume(role, data))
        print(f"  Generated {SAMPLES_PER_ROLE} resumes — {role}")

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)

    out_path = os.path.join(os.path.dirname(__file__), "structured_resumes.csv")
    df.to_csv(out_path, index=False)

    print(f"\nDataset saved: {out_path}")
    print(f"Total rows : {len(df)}")
    print(f"Unique roles: {df['job_role'].nunique()}")
    print(f"\nRole distribution:")
    print(df["job_role"].value_counts().to_string())
