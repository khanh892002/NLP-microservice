import re

# " surrounded by words/numbers:
SUSPICIOUS_QUOTE_REGEX = re.compile(r'([\w])"([\w])', re.UNICODE)
# " isolated by 2 space:
ISOLATED_QUOTE_REGEX = re.compile(r'\s"\s')

def check_suspicious_quotes(text: str) -> dict:
	if SUSPICIOUS_QUOTE_REGEX.search(text):
		return {
			"has_issue": True,
			"message": "double quotes surrounded by words/numbers."
		}
	if ISOLATED_QUOTE_REGEX.search(text):
		return {
			"has_issue": True,
			"message": "double quotes isolated by spaces will cause formatting error."
		}
	return {"has_issue": False, "message": None}


def convert_to_smart_quotes(text: str) -> str:
	# right at the beginning(^), space (\s), opening brackets, dash, dot
	text = re.sub(r'(^|[\s\(\[{\[\-_\.])"', r'\1“', text)
	# every other " are closing mark
	text = text.replace('"', '”')

	return text