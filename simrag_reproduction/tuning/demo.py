"""
Simple Tuning Demo
Basic testing for model fine-tuning functionality
"""

import os
from .basic_tuning import BasicTuner
from config import get_tuning_config


def run_tuning_demo(mode="quick"):
    """Run tuning demo in quick or full mode"""
    print("=== Tuning Demo ===")
    
    # Get configuration
    config = get_tuning_config()
    print(f"Using model: {config.model_name}")
    print(f"Device: {config.device}")
    print(f"Epochs: {config.num_epochs}")
    
    try:
        # Initialize tuner
        print("\nInitializing tuner...")
        tuner = BasicTuner(
            model_name=config.model_name,
            device=config.device
        )
        tuner.load_model()
        
        # Show model info
        info = tuner.get_model_info()
        print(f"Model loaded: {info['model_name']}")
        print(f"Parameters: {info['num_parameters']:,}")
        
        # Prepare sample data
        if mode == "quick":
            print("\n=== Quick Demo ===")
            training_texts = [
                "Machine learning is fascinating.",
                "Python is great for AI development.",
                "Deep learning models are powerful.",
                "Natural language processing is exciting.",
                "Computer vision helps machines see."
            ]
            epochs = 1
        else:
            print("\n=== Full Demo ===")
            training_texts = [
                "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming.",
                "Deep learning uses neural networks with multiple layers to model complex patterns in data.",
                "Natural language processing combines computational linguistics with machine learning to help computers understand human language.",
                "Computer vision enables machines to interpret and understand visual information from images and videos.",
                "Reinforcement learning is where agents learn optimal behavior by interacting with an environment and receiving rewards.",
                "Python is a versatile programming language widely used in data science and artificial intelligence.",
                "Neural networks are inspired by the structure and function of biological neural networks in the brain.",
                "Data preprocessing is crucial for successful machine learning model training and performance.",
                "Feature engineering involves selecting and transforming input variables to improve model accuracy.",
                "Model evaluation metrics help assess the performance and reliability of machine learning models."
            ]
            epochs = config.num_epochs
        
        print(f"Training with {len(training_texts)} examples for {epochs} epochs...")
        
        # Prepare data
        train_dataset = tuner.prepare_data(training_texts, max_length=config.max_length)
        
        # Setup trainer
        tuner.setup_trainer(
            train_dataset=train_dataset,
            output_dir=config.output_dir,
            num_epochs=epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate
        )
        
        # Train
        print("Starting training...")
        tuner.train()
        
        # Save model
        tuner.save_model(config.output_dir)
        print(f"Model saved to {config.output_dir}")
        
        # Test generation
        print("\n=== Testing Generation ===")
        test_prompts = [
            "Machine learning is",
            "Python is",
            "Deep learning"
        ]
        
        for prompt in test_prompts:
            try:
                generated = tuner.generate_text(prompt, max_length=50, temperature=0.7)
                print(f"'{prompt}' → '{generated}'")
            except Exception as e:
                print(f"Error generating for '{prompt}': {e}")
        
        print("\n✅ Tuning demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")


def main():
    """Main function for direct execution"""
    run_tuning_demo("quick")


if __name__ == "__main__":
    main()