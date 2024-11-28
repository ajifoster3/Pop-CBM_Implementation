prompts = dict(
    generator_system_prompt =
    '''
    You are an expert in the domain of heuristics. Your task is to design heuristics that can 
    effectively solve optimization problems. Your response outputs Python code and nothing else.
    Format your code as a Python code string : \"\''' python ... \'''\".
    ''',
    reflector_system_prompt =
    '''
    You are an expert in the domain of optimization heuristics. Your task is to give hints to design better heuristics.
    ''',
    task_description =
    '''
    Write a {function_name} function for {problem_description}
    {function_description}
    ''',
    user_prompt_population_initialisation =
    '''
    {task_description}
    
    {seed_function}
    
    Refer to the format of a trivial design above. Be very creative and give '{function_name}_v2'.
    Output code only and enclose your code with Python code block: \''' python ... \'''.
    
    {initial_longterm_reflection}
    ''',
    user_prompt_shortterm_reflection =
    '''
    Below are two {function_name} functions for {problem_description}
    {function_description}
    
    Your are provided with two code versions below, where the second version performs better than the first one.
    
    [Worse code]
    {worse_code}
    
    [Better code]
    {better_code}
    
    You respond with some hints for designing better heuristics, based on the two code versions and using less than
    20 words.
    ''',
    user_prompt_shortterm_reflection_on_blackbox_COPs =
    '''
    Below are two {function_name} functions for {problem_description}
    {function_description}
    
    Your are provided with two code versions below, where the second version performs better than the first one.
    
    [Worse code]
    {worse_code}
    
    [Better code]
    {better_code}
    
    Please infer the problem settings by comparing the two code versions and giving hints for designing better
    heuristics. You may give hints about how edge and node attributes correlate with the black-box objective value.
    Use less than 50 words.
    ''',
    user_prompt_crossover =
    '''
    {task_description}
    
    [Worse code]
    {function_signature0}
    {worse_code}
    
    [Better code]
    {function_signature1}
    {better_code}
    
    [Reflection]
    {shortterm_reflection}
    
    [improved code]
    Please write an improved function '{function_name}_v2', according to the reflection. Output code only and 
    enclose your code with Python code block: \''' python ... \'''.
    ''',
    user_prompt_longterm_reflection =
    '''
    Below is your prior long-term reflection on designing heuristics for {problem_description}
    {prior_longterm_reflection}
    
    Below are some newly gained insights.
    {new_shortterm_reflection}
    
    Write constructive hints for designing better heuristics, based on prior reflections and new insights and using 
    less than 50 words
    ''',
    user_prompt_elitist_mutation =
    '''
    {task_description}
    
    [prior reflection]
    {longterm_reflection}
    
    [Code]
    {function_signature1}
    {elitist_code}
    
    [Improved code]
    Please write a mutation function '{function_name}_v2', according to the reflection. Output code only and enclose 
    your code twith Python code block: \''' python ... \'''.
    '''
)