"""
Tuning Demo System
Interactive and automated testing for model fine-tuning
"""

import os
from typing import List, Optional
from .basic_tuning import BasicTuner
from config import TuningConfig, DEFAULT_CONFIGS


class TuningDemo:
    """Demo system for testing model fine-tuning"""
    
    def __init__(self, config: TuningConfig):
        """
        Initialize tuning demo
        
        Args:
            config: Tuning configuration
        """
        self.config = config
        self.tuner = None
    
    def initialize(self):
        """Initialize the tuner"""
        print("Initializing tuner...")
        
        # Create output directories
        self.config.create_output_dirs()
        
        self.tuner = BasicTuner(
            model_name=self.config.model_name,
            device=self.config.device
        )
        self.tuner.load_model()
        print("Tuner initialized successfully!")
    
    def load_sample_data(self) -> List[str]:
        """Load sample training data"""
        sample_texts = [
            "The weather is nice today. I think I'll go for a walk in the park.",
            "Machine learning is fascinating. I love studying neural networks and deep learning.",
            "Python is a great programming language for data science and AI development.",
            "The sunset was beautiful yesterday evening. The colors were amazing.",
            "I enjoy reading books about artificial intelligence and machine learning.",
            "Coffee helps me focus when I'm working on coding projects late at night.",
            "The mountains look amazing in the morning light. Nature is so peaceful.",
            "Deep learning models can solve complex problems that were impossible before.",
            "I like to take breaks and go outside for fresh air when coding.",
            "Natural language processing is an exciting field of study with many applications.",
            "The ocean waves are calming. I love listening to them on the beach.",
            "Programming requires patience and problem-solving skills to succeed.",
            "Artificial intelligence will change the world in many positive ways.",
            "Reading scientific papers helps me stay updated with the latest research.",
            "The forest is quiet and peaceful. Perfect for thinking and reflection."
        ]
        return sample_texts
    
    def load_custom_data(self, file_path: str) -> List[str]:
        """Load custom training data from file"""
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by lines or sentences
            texts = [line.strip() for line in content.split('\n') if line.strip()]
            return texts
        except Exception as e:
            print(f"Error loading file: {e}")
            return []
    
    def run_training(self, training_texts: List[str]):
        """Run the training process"""
        if not self.tuner:
            print("Error: Tuner not initialized")
            return
        
        print(f"\nTraining with {len(training_texts)} examples...")
        
        # Prepare data
        train_dataset = self.tuner.prepare_data(
            training_texts, 
            max_length=self.config.max_length
        )
        
        # Setup trainer
        self.tuner.setup_trainer(
            train_dataset=train_dataset,
            output_dir=self.config.output_dir,
            num_epochs=self.config.num_epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps
        )
        
        # Train
        self.tuner.train()
        
        # Save
        self.tuner.save_model(self.config.output_dir)
        print(f"Model saved to {self.config.output_dir}")
    
    def test_generation(self, prompts: List[str]):
        """Test text generation with the tuned model"""
        if not self.tuner:
            print("Error: Tuner not initialized")
            return
        
        print("\n=== Testing Text Generation ===")
        
        for prompt in prompts:
            try:
                generated = self.tuner.generate_text(
                    prompt,
                    max_length=self.config.max_generation_length,
                    temperature=self.config.temperature
                )
                print(f"\nPrompt: '{prompt}'")
                print(f"Generated: '{generated}'")
            except Exception as e:
                print(f"Error generating text for '{prompt}': {e}")
    
    def show_model_info(self):
        """Display model information"""
        if not self.tuner:
            print("Error: Tuner not initialized")
            return
        
        print("\n=== Model Information ===")
        info = self.tuner.get_model_info()
        for key, value in info.items():
            print(f"{key}: {value}")
    
    def run_full_demo(self):
        """Run complete demo"""
        self.config.print_config()
        
        # Initialize
        self.initialize()
        
        # Show model info
        self.show_model_info()
        
        # Load sample data
        training_texts = self.load_sample_data()
        print(f"\nLoaded {len(training_texts)} training examples")
        
        # Run training
        self.run_training(training_texts)
        
        # Test generation
        test_prompts = [
            "The weather is",
            "I love",
            "Python is",
            "Machine learning",
            "The sunset"
        ]
        self.test_generation(test_prompts)
    
    def run_quick_test(self):
        """Run a quick test with minimal training"""
        print("Running quick test...")
        
        # Use quick config
        quick_config = TuningConfig.for_quick_test()
        self.config = quick_config
        
        # Initialize
        self.initialize()
        
        # Load minimal data
        training_texts = self.load_sample_data()[:5]  # Just 5 examples
        
        # Run training
        self.run_training(training_texts)
        
        # Test generation
        self.test_generation(["The weather is", "I love"])


def main():
    """Main demo function"""
    from config import TuningConfig
    
    # Load configuration
    config = TuningConfig()
    
    # Create and run demo
    demo = TuningDemo(config)
    demo.run_full_demo()


if __name__ == "__main__":
    main()
