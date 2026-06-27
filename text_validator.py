def validate_brackets_and_quotes(text: str) -> dict:
	stack = []
	
	bracket_map = { ')': '(', ']': '[', '}': '{', '”': '“' }
	
	open_brackets = set(bracket_map.values())
	close_brackets = set(bracket_map.keys())
	
	for i, char in enumerate(text):
		if char in open_brackets:
			stack.append((char, i))
		elif char in close_brackets:
			if not stack:
				return {"is_valid": False, "error": f"Spare a closing mark'{char}' at index {i}."}
			
			last_open_char, _ = stack.pop()
			if bracket_map[char] != last_open_char:
				return {"is_valid": False, "error": f"Mismatch in nested structure: opened '{last_open_char}' but closed with '{char}' at index {i}."}
				
	if stack:
		unclosed_char, index = stack.pop()
		return {"is_valid": False, "error": f"Miss a closing mark for '{unclosed_char}' (opened at index {index})."}
		
	return {"is_valid": True, "error": None}