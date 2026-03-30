# Corporate & Political Data Pipeline (Goods Unite Us)

## Overview

This project implements an end-to-end **data engineering and NLP pipeline** designed to transform heterogeneous, real-world data sources into a unified, structured dataset for corporate political transparency analysis.

The system integrates multiple data sources—including **SEC filings, FEC political contribution data, Gmail user requests, and Snowflake company datasets**—to build a multi-layer dataset linking:

- Companies
- Executives
- Subsidiaries
- Political contributions

This project demonstrates applied skills in **data engineering, NLP, entity resolution, and large-scale data integration**.

---

## Key Features

- **Executive Extraction (SEC Filings)**
  - Parsed 10-K and 8-K filings to extract executive names and roles
  - Standardized and normalized executive titles across companies

- **Contact Enrichment**
  - Augmented company records with emails, domains, and metadata

- **Subsidiary Mapping**
  - Extracted parent–subsidiary relationships from SEC Exhibit 21 filings
  - Built structured corporate hierarchy datasets (~74K subsidiaries)

- **Gmail NLP Pipeline (Core Contribution)**
  - Parsed unstructured Gmail logs to extract brand/company requests
  - Implemented regex-based and rule-based text normalization
  - Generated structured datasets from noisy user-generated content

- **Firebase Integration**
  - Loaded processed datasets into a production-ready backend

- **PAC Data Integration**
  - Combined FEC and Snowflake datasets to classify company political activity

- **Entity Resolution & Ticker Matching**
  - Linked companies to stock tickers
  - Resolved executive ↔ company ↔ donation relationships

---

## Pipeline Architecture

The pipeline is composed of sequential stages:

1. **Extraction**
   - SEC filings (executives, subsidiaries)
   - Gmail logs (brand requests)

2. **Cleaning & Normalization**
   - Title standardization
   - Text preprocessing (regex + rule-based NLP)

3. **Enrichment**
   - Contact information
   - External metadata

4. **Integration**
   - FEC + Snowflake political data
   - Entity matching across datasets

5. **Output**
   - Final structured datasets for analysis and internal use

---
## Tech Stack

- **Python** (pandas, regex, data processing)  
- **NLP** (rule-based extraction, text normalization)  
- **SQL / Snowflake**  
- **Firebase**  
- **SEC EDGAR API**  
- **FEC datasets**

---

## Key Challenges

- Processing **highly unstructured text** (Gmail logs, SEC filings)  
- Resolving **entity ambiguity** (company names, executives)  
- Integrating **multiple heterogeneous data sources**  
- Handling **missing and inconsistent data**

---

## Key Achievements

- Built a **multi-source, end-to-end data pipeline**  
- Linked:
  - Companies ↔ Executives ↔ Subsidiaries ↔ Political Contributions  
- Developed reusable components for:
  - Text extraction  
  - Data cleaning  
  - Entity resolution  
- Produced structured datasets enabling **corporate political analysis**

---

## Project Significance

This project demonstrates the ability to design and implement **real-world data systems** that:

- Transform messy, unstructured data into usable datasets  
- Integrate multiple large-scale data sources  
- Support transparency and analytical insights in corporate behavior  

---

## Author

**Davo Acevedo-Cardona**  
MS in Human Language Technology — University of Arizona
