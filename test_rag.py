from rag_engine import get_rag_chain_manual, check_pii

def test_rbac():
    print("Testing RBAC...")
    
    # Test as Finance
    print("\n--- Role: Finance ---")
    finance_chain, _ = get_rag_chain_manual("finance")
    res = finance_chain.invoke("What are the marketing expenses for 2024?")
    print(f"Question: What are the marketing expenses for 2024?")
    print(f"Answer: {res}")
    
    # Test as HR
    print("\n--- Role: HR ---")
    hr_chain, _ = get_rag_chain_manual("hr")
    res = hr_chain.invoke("Can I see the marketing reports?")
    print(f"Question: Can I see the marketing reports?")
    print(f"Answer: {res}")

def test_guardrails():
    print("\nTesting Guardrails...")
    pii_query = "My email is test@example.com, can you help me?"
    if check_pii(pii_query):
        print(f"PII Detected in: {pii_query}")
    else:
        print(f"PII NOT Detected in: {pii_query} (Check regex)")

if __name__ == "__main__":
    test_guardrails()
    test_rbac()
