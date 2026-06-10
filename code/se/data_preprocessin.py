import random
import sqlite3
import pandas as pd
from typing import List, Dict

def load_dataset(db_path: str) -> List[Dict]:
    """Load data from SQLite database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name1, name2, interaction FROM event")
    data = []
    for name1, name2, interaction in cursor.fetchall():
        data.append({
            "drug1": name1,
            "drug2": name2,
            "interaction": interaction.strip()
        })
    
    conn.close()
    return data

def generate_prompts(data: List[Dict]) -> List[Dict]:
    """Generate diversified English prompt templates for drug-drug interactions"""
    prompt_templates = [
        # Direct inquiry
        "What is the interaction between {drug1} and {drug2}?",
        "Describe the drug-drug interaction between {drug1} and {drug2}.",
        "What happens when {drug1} is taken with {drug2}?",
        
        # Safety-related
        "Is it safe to combine {drug1} with {drug2}?",
        "What are the adverse effects of combining {drug1} and {drug2}?",
        "What are the risks of concomitant use of {drug1} and {drug2}?",
          
        # Mechanism-focused
        "How does {drug1} affect the pharmacological action of {drug2}?",
        "Explain the mechanism of interaction between {drug1} and {drug2}.",
        "How does {drug2} alter the metabolism of {drug1}?",
        
        # Clinical advice
        "What should patients know when taking both {drug1} and {drug2}?",
        "What are the clinical recommendations for combining {drug1} with {drug2}?",
        "How should clinicians manage the coadministration of {drug1} and {drug2}?",
        
        # Severity assessment
        "How severe is the interaction between {drug1} and {drug2}?",
        "What are the potential clinical consequences of {drug1}-{drug2} interaction?",
        "Evaluate the clinical significance of the interaction between {drug1} and {drug2}.",
        
        # Comparative
        "Compare the effects of {drug1} when used alone versus with {drug2}.",
        "How does the efficacy of {drug1} change when combined with {drug2}?",
        
        # Patient-specific
        "What should elderly patients know about taking {drug1} with {drug2}?",
        "Are there special considerations for renal patients taking {drug1} and {drug2}?"
    ]
    
    samples = []
    for item in data:
        selected_templates = random.sample(prompt_templates, k=random.randint(3, 5))
        
        for template in selected_templates:
            prompt = template.format(
                drug1=item["drug1"],
                drug2=item["drug2"]
            )
            
            samples.append({
                "prompt": prompt,
                "answer": item["interaction"],
                "drug1":item["drug1"],
                "drug2":item["drug2"]
            })
    
    return samples

def save_to_csv(samples: List[Dict], output_file: str) -> None:
    """Save processed data to CSV"""
    df = pd.DataFrame(samples)
    df.to_csv(output_file, index=False)
    print(f"Generated {len(df)} prompt-answer pairs, saved to {output_file}")

if __name__ == "__main__":
    db_path = "/data/cclsol/cfn/MF-Qwen/CD/dataset/event.db"
    output_file = "process_event.csv"
    
    try:
        raw_data = load_dataset(db_path)
        prompt_samples = generate_prompts(raw_data)
        save_to_csv(prompt_samples, output_file)
        
        print("\nSample prompt-answer pairs:")
        for sample in random.sample(prompt_samples, 3):
            print(f"Prompt: {sample['prompt']}")
            print(f"Answer: {sample['answer']}\n")
    except Exception as e:
        print(f"Error processing data: {e}")