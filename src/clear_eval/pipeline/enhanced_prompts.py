"""
Enhanced prompts for better handling of option-based tasks (PIQA, CommonsenseQA, SIQA, etc.)
Unified template with task-specific adaptations
"""

def is_option_based_response(response):
    """Check if response is just a single option (A, B, C, D, yes, no)"""
    response = response.strip().lower()
    # Check for single letter/number options
    if len(response) <= 3 and response.upper() in ['A', 'B', 'C', 'D', '0', '1']:
        return True
    # Check for yes/no responses (BoolQ)
    if response in ['yes', 'no', 'true', 'false']:
        return True
    return False

def detect_task_type(question):
    """Detect the type of multiple choice task based on question content"""
    question_lower = question.lower()

    if 'goal:' in question_lower and 'option a:' in question_lower:
        return 'piqa'  # Physical Interaction QA
    elif 'question:' in question_lower and ('option a:' in question_lower or 'a)' in question_lower):
        if any(word in question_lower for word in ['feel', 'emotion', 'social', 'person']):
            return 'siqa'  # Social Interaction QA
        else:
            return 'commonsenseqa'  # General CommonsenseQA
    elif 'passage:' in question_lower or 'context:' in question_lower:
        return 'reading_comprehension'
    elif any(word in question_lower for word in ['question:', 'answer:', 'true', 'false', 'yes', 'no']):
        # Check if it's a BoolQ-style task (passage + yes/no question)
        if 'passage' in question_lower or len(question.split()) > 50:
            return 'boolq'  # Boolean Question Answering

    return 'general_mcq'  # General multiple choice

def extract_options_from_question(question):
    """Extract options from various question formats"""
    options = {}
    lines = question.split('\n')

    # Pattern 1: "Option A:", "Option B:" format
    for line in lines:
        line = line.strip()
        if line.startswith('Option A:'):
            options['A'] = line.replace('Option A:', '').strip()
        elif line.startswith('Option B:'):
            options['B'] = line.replace('Option B:', '').strip()
        elif line.startswith('Option C:'):
            options['C'] = line.replace('Option C:', '').strip()
        elif line.startswith('Option D:'):
            options['D'] = line.replace('Option D:', '').strip()

    # Pattern 2: "A)", "B)" format
    if not options:
        for line in lines:
            line = line.strip()
            if line.startswith('A)') or line.startswith('a)'):
                options['A'] = line[2:].strip()
            elif line.startswith('B)') or line.startswith('b)'):
                options['B'] = line[2:].strip()
            elif line.startswith('C)') or line.startswith('c)'):
                options['C'] = line[2:].strip()
            elif line.startswith('D)') or line.startswith('d)'):
                options['D'] = line[2:].strip()

    return options

# Removed analyze_option_choice_quality() - GPT will handle this analysis

def get_unified_mcq_evaluation_prompt(model_input, model_output, reference, evaluation_criteria_str):
    """
    Unified evaluation prompt for all multiple-choice tasks with task-specific adaptations

    When --use-enhanced-mcq-evaluation=true is set, this function is called directly
    without checking if the response "looks like" an option (A/B/yes/no etc.)

    The user has explicitly requested MCQ evaluation, so we provide it regardless of output format.
    """

    # Detect task type and adapt accordingly
    task_type = detect_task_type(model_input)

    # Simple correctness check - let GPT handle detailed analysis
    is_correct = model_output.strip().lower() == reference.strip().lower()

    # Task-specific context and criteria
    task_contexts = {
            'piqa': {
                'name': 'Physical Interaction QA (PIQA)',
                'description': 'physical reasoning and practical problem-solving',
                'key_aspects': ['physical principles', 'safety awareness', 'practical feasibility'],
                'common_errors': ['safety misjudgment', 'physics misconception', 'impractical solution']
            },
            'siqa': {
                'name': 'Social Interaction QA (SIQA)',
                'description': 'social reasoning and interpersonal understanding',
                'key_aspects': ['social norms', 'emotional intelligence', 'interpersonal dynamics'],
                'common_errors': ['social inappropriateness', 'emotional insensitivity', 'context misreading']
            },
            'commonsenseqa': {
                'name': 'CommonsenseQA',
                'description': 'general commonsense reasoning',
                'key_aspects': ['logical reasoning', 'world knowledge', 'common sense'],
                'common_errors': ['logical fallacy', 'knowledge gap', 'overthinking simple concepts']
            },
            'boolq': {
                'name': 'Boolean Question Answering (BoolQ)',
                'description': 'reading comprehension and factual reasoning',
                'key_aspects': ['reading comprehension', 'factual accuracy', 'logical inference'],
                'common_errors': ['misreading passage', 'factual misinterpretation', 'logical inference error']
            },
            'general_mcq': {
                'name': 'Multiple Choice Question',
                'description': 'general reasoning and comprehension',
                'key_aspects': ['comprehension', 'logical reasoning', 'knowledge application'],
                'common_errors': ['miscomprehension', 'logical error', 'knowledge misapplication']
            }
    }

    context = task_contexts.get(task_type, task_contexts['general_mcq'])

    return f"""You are an expert evaluator of AI models on multiple-choice reasoning tasks. You will evaluate a model's option choice and provide detailed feedback.

Task Context: This is a {context['name']} task focusing on {context['description']}.

Input Question: {model_input}
Model's Choice: {model_output}
Correct Answer: {reference}
Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}

For option-based responses, evaluate based on:

1. **Correctness**: Did the model choose the right option?
2. **Reasoning Quality** (inferred from choice pattern):
   - Key aspects for this task: {', '.join(context['key_aspects'])}
   - Does the choice demonstrate understanding of these aspects?
3. **Error Analysis** (if incorrect):
   - What type of error does this represent?
   - Common error categories: {', '.join(context['common_errors'])}
   - What does this choice reveal about the model's reasoning process?

Scoring Guidelines for Option-Based Tasks:
- 1.0: Correct choice demonstrating strong understanding of task requirements
- 0.8: Correct choice with solid reasoning
- 0.6: Incorrect choice but shows partial understanding of key concepts
- 0.4: Incorrect choice with clear reasoning error in task domain
- 0.2: Incorrect choice showing fundamental misunderstanding
- 0.0: Completely irrelevant or nonsensical choice

Please analyze what this choice reveals about the model's understanding of {context['description']} and provide specific insights into the reasoning quality.

Evaluation Criteria:
{evaluation_criteria_str}

--- Begin Evaluation ---
Textual Evaluation: [Your detailed analysis of the choice and reasoning quality]
Evaluation score: [Your score here]"""

def get_enhanced_shortcomings_synthesis_prompt(concatenated_evaluation_texts: str, max_shortcomings: int):
    """Enhanced shortcomings synthesis that considers option-based task patterns"""
    
    return f"""You are an expert analyst specializing in evaluating AI models on multiple-choice reasoning tasks. Below is a collection of evaluation texts assessing the quality of model responses to physical reasoning questions.

Your goal is to identify the most significant and frequent types of shortcomings, with special attention to patterns common in option-based tasks. Please provide a list of up to {max_shortcomings} concise phrases describing these common issues.

Focus on these categories of shortcomings:

**Choice-Making Issues:**
- Incorrect option selection patterns
- Lack of reasoning for choices
- Opposite logic errors (choosing negation of correct answer)
- Safety misjudgments in physical scenarios

**Reasoning Quality Issues:**
- Misunderstanding of physical principles
- Contextual misinterpretation
- Oversimplification of complex scenarios
- Failure to consider practical constraints

**Response Quality Issues:**
- Insufficient explanation or justification
- Factual inaccuracies about physical processes
- Incomplete analysis of options
- Poor alignment with task requirements

Provide actionable feedback points that could help improve the model's performance on similar reasoning tasks.

Evaluation Texts:
--- Begin Evaluation Texts ---
{concatenated_evaluation_texts}
--- End Evaluation Texts ---

Synthesized List of Shortcomings (Python List format ONLY):
"""

# Removed categorize_failure_type() - GPT will handle error categorization through natural language analysis
