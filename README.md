# Salesforce Lead Scoring Prototype

This project builds a simple lead scoring model in Python using a public lead-conversion dataset. The script downloads the dataset, preprocesses the features, compares two classification models, and uses the stronger model to score new leads.

## Files

- `salesforce_lead_scoring.py` - main modeling and scoring script
- `requirements.txt` - Python dependencies
- `data/` - cached public dataset
- `outputs/` - saved metrics and sample scoring results

## Public dataset

- Source CSV: https://raw.githubusercontent.com/wcrowley342/LogisticRegression/main/Lead%20Source%20Dataset%20V14.csv
- Source repository: https://github.com/wcrowley342/LogisticRegression

The public file includes:

- `LeadSource`
- `CompanySize`
- `WebsiteVisits`
- `EmailOpens`
- `EmailClicks`
- conversion outcome in `Customer`

It does not include an explicit `Industry` column, so the prototype uses the available firmographic and engagement fields directly.

## Run locally

```bash
pip install -r requirements.txt
python salesforce_lead_scoring.py
```
