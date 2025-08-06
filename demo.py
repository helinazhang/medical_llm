#!/usr/bin/env python3
"""
Simple Medical AI Demo
Error-free Streamlit interface for testing medical AI
"""

import streamlit as st
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
current_dir = Path(__file__).parent.absolute()
src_dir = current_dir.parent / "src" if (current_dir.parent / "src").exists() else current_dir / "src"
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

def load_model_safely(model_path):
    """Safely load model or return None"""
    try:
        # Try to import and load model
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

def generate_response(model, tokenizer, question, max_length=200):
    """Generate response from model"""
    try:
        # Format input
        input_text = f"Question: {question}\nAnswer:"
        
        # Tokenize
        inputs = tokenizer.encode(input_text, return_tensors="pt")
        
        # Generate
        with st.spinner("AI is thinking..."):
            outputs = model.generate(
                inputs,
                max_length=len(inputs[0]) + max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.replace(input_text, "").strip()
        
        # Clean up answer
        if "Question:" in answer:
            answer = answer.split("Question:")[0].strip()
        if "\n" in answer:
            answer = answer.split("\n")[0].strip()
            
        return answer if answer else "I'm not sure how to answer that question."
        
    except Exception as e:
        return f"Error generating response: {e}"

def main():
    """Main Streamlit app"""
    
    # Page config
    st.set_page_config(
        page_title="Medical AI Demo",
        layout="wide"
    )
    
    # Title
    st.title("Medical AI Demo")
    st.markdown("Medical AI Model")
    
    # Sidebar
    with st.sidebar:
        st.header("Model Settings")
        
        # Model path input
        model_path = st.text_input(
            "Model Path:",
            value="models/medical-exam-tutor",
            help="Path to your trained model"
        )
        
        # Load model button
        load_model = st.button("Load Model")
        
        # Model status
        if "model_loaded" not in st.session_state:
            st.session_state.model_loaded = False
            st.session_state.model = None
            st.session_state.tokenizer = None
        
        if load_model:
            st.session_state.model, st.session_state.tokenizer = load_model_safely(model_path)
            st.session_state.model_loaded = (st.session_state.model is not None)
        
        # Status indicator
        if st.session_state.model_loaded:
            st.success("Model loaded successfully!")
        else:
            st.warning("No model loaded")
            
        st.divider()
        
        # Example questions
        st.header("Example Questions")
        
        examples = [
            "What are the symptoms of diabetes?",
            "How does the heart work?",
            "What causes high blood pressure?",
            "How do ACE inhibitors work?",
            "What is insulin resistance?",
            "Signs of a heart attack?",
            "How is pneumonia treated?",
            "What causes asthma?"
        ]
        
        st.write("Click to copy to clipboard:")
        for i, example in enumerate(examples):
            if st.button(f"{example[:30]}...", key=f"ex_{i}"):
                st.code(example)
                st.info("Copy this question!")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Ask Your Medical Question")
        
        # Question input
        question = st.text_area(
            "Enter your medical question:",
            height=120,
            placeholder="Type your medical question here...\n\nExample: What are the symptoms of diabetes?"
        )
        
        # Generate button
        if st.button("Ask AI", type="primary", disabled=not question.strip()):
            if not st.session_state.model_loaded:
                st.error("Please load a model first!")
            elif question.strip():
                # Generate response
                response = generate_response(
                    st.session_state.model,
                    st.session_state.tokenizer,
                    question.strip()
                )
                
                # Display response
                st.subheader("AI Response:")
                st.write(response)
                
                # Save to history
                if "conversation_history" not in st.session_state:
                    st.session_state.conversation_history = []
                
                st.session_state.conversation_history.append({
                    "question": question.strip(),
                    "answer": response,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
        
        # Test without model
        st.divider()
        st.subheader("Test Without Model")
        if st.button("Test Demo Response"):
            test_response = """
            **Diabetes symptoms typically include:**
            
            1. **Increased thirst** (polydipsia)
            2. **Frequent urination** (polyuria)  
            3. **Unexplained weight loss**
            4. **Fatigue and weakness**
            5. **Blurred vision**
            6. **Slow-healing wounds**
            7. **Increased hunger** (polyphagia)
            
            If you experience these symptoms, consult a healthcare provider for proper diagnosis and testing.
            """
            st.markdown(test_response)
    
    with col2:
        st.subheader("Quick Stats")
        
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []
        
        total_questions = len(st.session_state.conversation_history)
        st.metric("Questions Asked", total_questions)
        
        if st.session_state.model_loaded:
            st.metric("Model Status", "Loaded")
        else:
            st.metric("Model Status", "Not Loaded")
        
        # Clear history
        if st.button("Clear History"):
            st.session_state.conversation_history = []
            st.success("History cleared!")
        
        st.subheader("Recent Questions")
        
        # Show recent questions
        for i, item in enumerate(reversed(st.session_state.conversation_history[-5:])):
            with st.expander(f"Q{len(st.session_state.conversation_history)-i}: {item['question'][:30]}..."):
                st.write(f"**Question:** {item['question']}")
                st.write(f"**Answer:** {item['answer']}")
                st.write(f"**Time:** {item['timestamp']}")

if __name__ == "__main__":
    main()
