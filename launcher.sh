#!/bin/bash

# Launcher script for Hybrid Clinical Decision Support System

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║   Hybrid Neuro-Symbolic Clinical Decision Support System        ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Detect Python command
if [ -f "/home/dev/projects/python/vspython/.venv/bin/python" ]; then
    PYTHON_CMD="/home/dev/projects/python/vspython/.venv/bin/python"
    echo "✓ Using virtual environment"
else
    PYTHON_CMD="python3"
    echo "✓ Using system Python"
fi

echo ""
echo "What would you like to do?"
echo ""
echo "  1) Launch GUI (Web Interface)"
echo "  2) Train CNN (with auto-resume)"
echo "  3) Train All Models"
echo "  4) Test System"
echo "  5) Monitor Training Progress"
echo "  6) Exit"
echo ""
read -p "Enter choice [1-6]: " choice

case $choice in
    1)
        echo ""
        echo "🚀 Launching Web Interface..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Opening at: http://localhost:7860"
        echo "Press Ctrl+C to stop"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        $PYTHON_CMD app.py
        ;;
    
    2)
        echo ""
        echo "🔄 Training CNN with auto-resume..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # Check for existing checkpoint
        if [ -f "models/cnn_skin_lesion_checkpoint.pth" ]; then
            echo "✓ Found checkpoint - will resume training"
        else
            echo "• No checkpoint found - starting fresh"
        fi
        
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        $PYTHON_CMD train.py --train-cnn 2>&1 | tee cnn_training_log.txt
        ;;
    
    3)
        echo ""
        echo "🎓 Training all models..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        $PYTHON_CMD train.py --train-all
        ;;
    
    4)
        echo ""
        echo "🧪 Testing system..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        $PYTHON_CMD main.py --patient-file examples/covid19_patient.json
        ;;
    
    5)
        echo ""
        echo "📊 Monitoring training progress..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Press Ctrl+C to stop monitoring"
        echo ""
        
        if [ -f "cnn_training_log.txt" ]; then
            tail -f cnn_training_log.txt
        else
            echo "❌ No training log found (cnn_training_log.txt)"
            echo "Start training first with option 2"
        fi
        ;;
    
    6)
        echo "Goodbye!"
        exit 0
        ;;
    
    *)
        echo "Invalid choice. Please run again and select 1-6."
        exit 1
        ;;
esac
