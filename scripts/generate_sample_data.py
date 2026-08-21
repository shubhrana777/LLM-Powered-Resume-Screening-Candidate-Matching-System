"""Generate synthetic sample resumes for local development and demos.

Every candidate here is fictional. Real resumes contain personal data and must
never be committed, so ``data/resumes/`` is git-ignored and this script
regenerates its contents on demand:

    python scripts/generate_sample_data.py

The set is deliberately varied -- backend, ML, data, frontend, DevOps, and one
completely unrelated profile -- so that the Phase 2 ranking can be sanity
checked: relevant candidates should sort above the unrelated one.
"""

from __future__ import annotations

from pathlib import Path

import fitz

RESUME_DIR = Path("data/resumes")

# (filename stem, [lines]) -- all candidates are invented.
SAMPLE_RESUMES: dict[str, list[str]] = {
    "priya_sharma": [
        "Priya Sharma",
        "Senior Python Backend Engineer",
        "priya.sharma@example.com | Bengaluru, India",
        "",
        "SUMMARY",
        "Backend engineer with 8 years building scalable Python microservices",
        "and REST APIs on AWS. Strong focus on distributed systems and",
        "database performance.",
        "",
        "EXPERIENCE",
        "Lead Backend Engineer, Nimbus Data (2021-2025)",
        "Designed and shipped Python FastAPI microservices handling 5M",
        "requests per day. Reduced p99 latency by 45 percent through",
        "PostgreSQL query tuning and Redis caching.",
        "Mentored four engineers and owned the service reliability roadmap.",
        "",
        "Backend Engineer, Orbit Systems (2017-2021)",
        "Built asynchronous task pipelines with Celery and RabbitMQ.",
        "Migrated a monolith to containerized services on Kubernetes.",
        "",
        "SKILLS",
        "Python, FastAPI, Django, PostgreSQL, Redis, Docker, Kubernetes,",
        "AWS, REST API design, distributed systems, SQL, CI/CD",
        "",
        "EDUCATION",
        "B.Tech Computer Science, Indian Institute of Technology (2017)",
    ],
    "marcus_chen": [
        "Marcus Chen",
        "Machine Learning Engineer",
        "marcus.chen@example.com | Toronto, Canada",
        "",
        "SUMMARY",
        "ML engineer specialising in natural language processing and",
        "recommendation systems. Comfortable taking models from notebook",
        "to production Python services.",
        "",
        "EXPERIENCE",
        "Machine Learning Engineer, Vector Labs (2020-2025)",
        "Built transformer-based text classification models with PyTorch",
        "and Hugging Face. Deployed inference services in Python and Docker.",
        "Developed a semantic search system using sentence embeddings and",
        "a FAISS vector index over 2M documents.",
        "",
        "Data Scientist, Northwind Analytics (2018-2020)",
        "Trained gradient boosting models for churn prediction in Python.",
        "",
        "SKILLS",
        "Python, PyTorch, Hugging Face, transformers, NLP, embeddings,",
        "FAISS, scikit-learn, Docker, SQL, REST APIs",
        "",
        "EDUCATION",
        "M.Sc. Computer Science, University of Toronto (2018)",
    ],
    "david_kim": [
        "David Kim",
        "DevOps / Site Reliability Engineer",
        "david.kim@example.com | Seoul, South Korea",
        "",
        "SUMMARY",
        "Infrastructure engineer focused on Kubernetes, observability and",
        "build automation for Python and Go backend services.",
        "",
        "EXPERIENCE",
        "Site Reliability Engineer, Helios Cloud (2019-2025)",
        "Ran multi-region Kubernetes clusters serving containerized backend",
        "APIs. Automated CI/CD with GitHub Actions and Terraform.",
        "Cut deployment times by 60 percent and built the on-call alerting",
        "stack on Prometheus and Grafana.",
        "",
        "Systems Engineer, Daylight Networks (2016-2019)",
        "Managed Linux servers and wrote Python automation tooling.",
        "",
        "SKILLS",
        "Kubernetes, Docker, Terraform, AWS, Linux, Python scripting,",
        "Prometheus, Grafana, CI/CD, PostgreSQL administration",
        "",
        "EDUCATION",
        "B.Eng Information Systems, Korea University (2016)",
    ],
    "elena_rodriguez": [
        "Elena Rodriguez",
        "Data Analyst",
        "elena.rodriguez@example.com | Madrid, Spain",
        "",
        "SUMMARY",
        "Data analyst turning business questions into dashboards and",
        "reports. Strong SQL, comfortable with light Python scripting.",
        "",
        "EXPERIENCE",
        "Senior Data Analyst, Mercado Insights (2020-2025)",
        "Built executive dashboards in Tableau and Power BI.",
        "Wrote complex SQL against a Snowflake warehouse to model revenue.",
        "Automated weekly reporting with Python and pandas.",
        "",
        "Data Analyst, Cielo Retail (2018-2020)",
        "Performed cohort and funnel analysis for an e-commerce business.",
        "",
        "SKILLS",
        "SQL, Tableau, Power BI, Excel, pandas, statistics, data modelling,",
        "Snowflake, reporting, A/B testing",
        "",
        "EDUCATION",
        "B.A. Economics, Universidad Complutense de Madrid (2018)",
    ],
    "aisha_okafor": [
        "Aisha Okafor",
        "Frontend Engineer",
        "aisha.okafor@example.com | Lagos, Nigeria",
        "",
        "SUMMARY",
        "Frontend engineer building accessible React interfaces and design",
        "systems for consumer web products.",
        "",
        "EXPERIENCE",
        "Senior Frontend Engineer, Bright Interfaces (2021-2025)",
        "Led the migration of a large React codebase to TypeScript.",
        "Built a reusable component library used by six product teams.",
        "Improved Lighthouse accessibility scores from 68 to 97.",
        "",
        "Frontend Developer, Kola Digital (2018-2021)",
        "Developed responsive marketing sites with JavaScript and CSS.",
        "",
        "SKILLS",
        "React, TypeScript, JavaScript, HTML, CSS, Next.js, accessibility,",
        "design systems, Jest, Webpack, UI testing",
        "",
        "EDUCATION",
        "B.Sc. Computer Science, University of Lagos (2018)",
    ],
    # --- Finance candidates, used to demonstrate Phase 3 skill analysis ------
    # sarah_wilson is a strong match, james_patel a partial one, and
    # nina_volkov a poor one that also states no years of experience.
    "sarah_wilson": [
        "Sarah Wilson",
        "Senior Financial Analyst",
        "sarah.wilson@example.com | Manchester, United Kingdom",
        "",
        "SUMMARY",
        "Financial analyst with 4 years of experience in budgeting,",
        "forecasting and management reporting for mid-sized businesses.",
        "",
        "EXPERIENCE",
        "Senior Financial Analyst, Northgate Retail (2022-2025)",
        "Built rolling forecasts and financial modeling for a 40M revenue unit.",
        "Automated monthly management reporting using Python and SQL,",
        "cutting the close cycle from nine days to four.",
        "Developed Power BI dashboards used by the executive team.",
        "",
        "Financial Analyst, Kestrel Group (2021-2022)",
        "Produced variance analysis and supported budget planning.",
        "Performed risk analysis on supplier exposure.",
        "",
        "SKILLS",
        "Excel, financial modeling, forecasting, SQL, Python, Power BI,",
        "Tableau, budgeting, risk analysis, data analysis, statistics",
        "",
        "EDUCATION",
        "MBA in Finance, Manchester Business School (2021)",
        "B.Com Accounting, University of Leeds (2018)",
    ],
    "james_patel": [
        "James Patel",
        "Junior Finance Associate",
        "james.patel@example.com | Birmingham, United Kingdom",
        "",
        "SUMMARY",
        "Finance associate with 2 years of experience supporting month-end",
        "reporting and reconciliations.",
        "",
        "EXPERIENCE",
        "Finance Associate, Ridgeway Logistics (2023-2025)",
        "Prepared monthly reconciliations and assisted with budgeting.",
        "Maintained reporting workbooks in Excel and ran SQL queries",
        "against the finance data warehouse.",
        "",
        "SKILLS",
        "Excel, SQL, budgeting, accounting, reporting, attention to detail",
        "",
        "EDUCATION",
        "B.Com Accounting and Finance, Aston University (2023)",
    ],
    "nina_volkov": [
        "Nina Volkov",
        "Graphic Designer",
        "nina.volkov@example.com | Prague, Czech Republic",
        "",
        "SUMMARY",
        "Graphic designer focused on brand identity, packaging and print.",
        "",
        "EXPERIENCE",
        "Senior Graphic Designer, Studio Vlna",
        "Led brand identity projects for food and beverage clients.",
        "Art directed packaging ranges and produced print-ready artwork.",
        "",
        "Graphic Designer, Prisma Creative",
        "Designed posters, brochures and social campaigns.",
        "",
        "SKILLS",
        "Illustrator, Photoshop, InDesign, typography, branding,",
        "packaging design, print production, art direction",
        "",
        "EDUCATION",
        "Diploma in Graphic Design, Prague College of Art",
    ],
    "tom_baker": [
        "Tom Baker",
        "Executive Pastry Chef",
        "tom.baker@example.com | Bristol, United Kingdom",
        "",
        "SUMMARY",
        "Pastry chef with 15 years in fine dining. Menu development,",
        "kitchen leadership, and seasonal dessert programmes.",
        "",
        "EXPERIENCE",
        "Executive Pastry Chef, The Harbour Room (2016-2025)",
        "Led a brigade of nine in a Michelin-recommended restaurant.",
        "Created seasonal tasting menus and managed supplier relationships.",
        "Reduced ingredient waste by 30 percent through portion planning.",
        "",
        "Pastry Sous Chef, Willow & Rye (2010-2016)",
        "Produced breads, viennoiserie and plated desserts daily.",
        "",
        "SKILLS",
        "Pastry, baking, chocolate work, sugar craft, menu design,",
        "kitchen management, food safety, inventory control, catering",
        "",
        "EDUCATION",
        "Diploma in Professional Patisserie, Bath College (2010)",
    ],
}


def write_resume_pdf(path: Path, lines: list[str]) -> Path:
    """Write ``lines`` to a single-page PDF at ``path``.

    Args:
        path: Destination PDF path.
        lines: Resume lines; empty strings become vertical spacing.

    Returns:
        The path written.
    """
    document = fitz.open()
    try:
        page = document.new_page()
        y = 60
        for position, line in enumerate(lines):
            size = 15 if position == 0 else 10
            if line:
                page.insert_text((60, y), line, fontsize=size)
            y += size + 5
        document.save(path)
    finally:
        document.close()
    return path


def main() -> int:
    """Write every sample resume into :data:`RESUME_DIR`."""
    RESUME_DIR.mkdir(parents=True, exist_ok=True)

    for stem, lines in sorted(SAMPLE_RESUMES.items()):
        path = write_resume_pdf(RESUME_DIR / f"{stem}.pdf", lines)
        print(f"wrote {path}")

    print(f"\n{len(SAMPLE_RESUMES)} sample resumes written to {RESUME_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
