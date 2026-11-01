import os
from litellm import completion

def verify_provider(model_name, api_key, provider_label):
    """
    Executes a structured connection probe using dot-notation property 
    lookups to match LiteLLM's internal choices response object.
    """
    if not api_key or not api_key.strip():
        print(f"[!] {provider_label} SKIPPED: Missing or empty API key environment token.")
        return False

    print(f"[*] Probing {provider_label} Substrate via '{model_name}'... (Key length: {len(api_key.strip())})")
    
    try:
        response = completion(
            model=model_name,
            messages=[{"role": "user", "content": "Respond with the word: operational"}],
            api_key=api_key.strip(),
            timeout=8.0  # Safe headroom for remote gateway routers
        )
        
        # Standard LiteLLM response structure parsing using dot-notation
        content = response.choices[0].message.content.strip()
        print(f"[+] {provider_label} SUCCESS: \"{content}\"\n")
        return True
    except Exception as e:
        print(f"[-] {provider_label} FAILURE: {str(e)}\n")
        return False

def main():
    print("=================================================================")
    print("      HARI CORE PLATFORM INFRASTRUCTURE SUBSTRATE SUITE          ")
    print("=================================================================\n")
    
    # 1. Google Gemini Endpoint
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    verify_provider("gemini/gemini-2.5-flash", gemini_key, "GOOGLE GEMINI")
    
    # 2. Groq Production Frontier Model
    groq_key = os.getenv("GROQ_API_KEY")
    verify_provider("groq/openai/gpt-oss-120b", groq_key, "GROQ LPU ADVANCED")
    
    # 3. Groq Production Light Model
    verify_provider("groq/openai/gpt-oss-20b", groq_key, "GROQ LPU LIGHT")

    # 4. Mistral Edge Endpoint 
    mistral_key = os.getenv("MISTRAL_API_KEY")
    verify_provider("mistral/mistral-small-latest", mistral_key, "MISTRAL NATIVE")

    # 5. OpenRouter Multi-Gateway Router
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    verify_provider("openrouter/meta-llama/llama-3.3-70b-instruct", openrouter_key, "OPENROUTER GATEWAY")

if __name__ == "__main__":
    main()
