import sys
import spacy
import hashlib
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langdetect import detect, LangDetectException
from text_validator import validate_brackets_and_quotes
from quotes_sanitizer import check_suspicious_quotes, convert_to_smart_quotes

app = FastAPI(
	title="Sentence Analyzer API",
	version="1.0.0",
	description="NLP sentence structure analysis powered by spaCy."
)

# Versioned router — all new endpoints go here
api_v1 = APIRouter(prefix="/api/v1")

# SECURITY FIX: Restrict CORS
# Update these domains with your actual production frontend URL
ALLOWED_ORIGINS = [
	"http://localhost:5173", # Vite dev
	"http://localhost:3000", # Node proxy
	"http://localhost:4173", # Vite preview
	"https://syntax-analyzer-24163.firebaseapp.com",
	"https://syntax-analyzer-24163.web.app",
]

app.add_middleware(
	CORSMiddleware,
	allow_origins=ALLOWED_ORIGINS,
	allow_credentials=True,
	allow_methods=["GET", "POST", "OPTIONS"],
	allow_headers=["*"],
)

# Initialize Firebase Admin SDK for Global Cache
# This requires serviceAccountKey.json in the project root
SA_KEY_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'serviceAccountKey.json')
db = None

if os.path.exists(SA_KEY_PATH):
	try:
		cred = credentials.Certificate(SA_KEY_PATH)
		firebase_admin.initialize_app(cred)
		db = firestore.client()
		print("Firebase Admin SDK initialized successfully.")
	except Exception as e:
		print(f"Error initializing Firebase Admin: {e}")
else:
	print(f"WARNING: serviceAccountKey.json not found at {SA_KEY_PATH}. Global Cache will be disabled.")

try:
	# Load the English language model
	nlp = spacy.load("en_core_web_sm")
except OSError:
	print("Model en_core_web_sm not found. Please run: python -m spacy download en_core_web_sm")
	sys.exit(1)

class SentenceRequest(BaseModel):
	sentence: str

def build_tree(token):
	node = {
		"role": token.dep_,
		"type": "word",
		"text": token.text,
		"pos": token.pos_
	}
	
	children = list(token.children)
	
	if children:
		node["type"] = "phrase"
		content = []
		
		all_tokens = children + [token]
		all_tokens.sort(key=lambda x: x.i)
		
		for child in all_tokens:
			if child == token:
				content.append({
					"role": "head",
					"type": "word",
					"text": child.text,
					"pos": child.pos_
				})
			else:
				content.append(build_tree(child))
				
		node["content"] = content
		if "text" in node:
			del node["text"]
			
	return node

def find_top_level_brackets(text: str) -> list:
	bracket_map = { ')': '(', ']': '[', '}': '{', '”': '“' }
	open_brackets = set(bracket_map.values())
	close_brackets = set(bracket_map.keys())
	
	spans = []
	stack = []
	
	for i, char in enumerate(text):
		if char in open_brackets:
			stack.append((char, i))
		elif char in close_brackets:
			if stack:
				open_char, start_idx = stack.pop()
				if not stack:
					spans.append((start_idx, i, open_char, char))
	return spans

def analyze_sentence_recursive(text: str):
	spans = find_top_level_brackets(text)
	if not spans:
		doc = nlp(text)
		sents = list(doc.sents)
		if not sents:
			return None
		sent = sents[0]
		roots = [token for token in sent if token.dep_ == "ROOT"]
		if not roots:
			return None
		return build_tree(roots[0])
		
	# Reverse sort spans by start_idx to replace from right to left
	spans.sort(key=lambda x: x[0], reverse=True)
	placeholders = []
	modified_text = text
	
	for idx, (start_idx, end_idx, open_char, close_char) in enumerate(spans):
		placeholder = f"somethingX{idx}"
		inner_text = text[start_idx + 1:end_idx].strip()
		
		# Recursively parse the inner text
		inner_tree = analyze_sentence_recursive(inner_text)
		
		placeholders.append({
			"placeholder": placeholder,
			"open_char": open_char,
			"close_char": close_char,
			"inner_tree": inner_tree
		})
		
		modified_text = modified_text[:start_idx] + placeholder + modified_text[end_idx + 1:]
		
	doc = nlp(modified_text)
	sents = list(doc.sents)
	if not sents:
		return None
	sent = sents[0]
	roots = [token for token in sent if token.dep_ == "ROOT"]
	if not roots:
		return None
		
	tree = build_tree(roots[0])
	
	# Check if root itself is a placeholder
	root_text = tree.get("text", "")
	for p in placeholders:
		if root_text == p["placeholder"]:
			bracket_group = {
				"role": tree.get("role", "ROOT"),
				"type": "bracket_group",
				"pos": "PUNCT",
				"content": [
					{
						"role": "punct",
						"type": "word",
						"text": p["open_char"],
						"pos": "PUNCT"
					}
				]
			}
			if p["inner_tree"]:
				bracket_group["content"].append(p["inner_tree"])
			bracket_group["content"].append({
				"role": "punct",
				"type": "word",
				"text": p["close_char"],
				"pos": "PUNCT"
			})
			return bracket_group
			
	def replace_placeholders(node):
		if node.get("type") == "phrase" and "content" in node:
			new_content = []
			for child in node["content"]:
				found_placeholder = None
				child_text = child.get("text", "")
				for p in placeholders:
					if child_text == p["placeholder"]:
						found_placeholder = p
						break
				if found_placeholder:
					bracket_group = {
						"role": child.get("role", "dep"),
						"type": "bracket_group",
						"pos": "PUNCT",
						"content": [
							{
								"role": "punct",
								"type": "word",
								"text": found_placeholder["open_char"],
								"pos": "PUNCT"
							}
						]
					}
					if found_placeholder["inner_tree"]:
						bracket_group["content"].append(found_placeholder["inner_tree"])
					bracket_group["content"].append({
						"role": "punct",
						"type": "word",
						"text": found_placeholder["close_char"],
						"pos": "PUNCT"
					})
					new_content.append(bracket_group)
				else:
					replace_placeholders(child)
					new_content.append(child)
			node["content"] = new_content
			
	replace_placeholders(tree)
	return tree

def get_sentence_hash(text: str) -> str:
	return hashlib.sha256(text.encode('utf-8')).hexdigest()

@api_v1.post("/analyze", summary="Analyze sentence structure")
def analyze_sentence(request: SentenceRequest):
	text = request.sentence
	
	if not text or not text.strip():
		raise HTTPException(status_code=400, detail="Sentence input is required.")
	
	if len(text) > 5000:
		raise HTTPException(status_code=400, detail="The text is too long. Please keep the input text 5000 characters at most.")
		
	try:
		lang = detect(text)
		if lang != 'en':
			raise HTTPException(status_code=400, detail=f"Only use English, please! ({lang} detected).")
	except LangDetectException:
		raise HTTPException(status_code=400, detail="Gibberish or unintelligible.")
	
	
	even_double_quotes = True
	for c in text:
		if c == '"': even_double_quotes ^= 1

	if not even_double_quotes:
		raise HTTPException(status_code=400, detail="The number of double quotes is odd.")

	suspicious_quotes = check_suspicious_quotes(text)
	if suspicious_quotes["has_issue"]:
		raise HTTPException(status_code=400, detail=suspicious_quotes["message"])

	text = convert_to_smart_quotes(text)

	validation_result = validate_brackets_and_quotes(text)
	if not validation_result["is_valid"]:
		raise HTTPException(status_code=400, detail=validation_result["error"])
	
	doc = nlp(text)
	
	results = []
	for sent in doc.sents:
		sent_text = sent.text.strip()
		if not sent_text:
			continue
			
		sent_hash = get_sentence_hash(sent_text)
		cached_tree = None
		
		# 1. Check Global Cache in Firestore
		if db:
			doc_ref = db.collection('global_sentence_cache').document(sent_hash)
			doc_snap = doc_ref.get()
			if doc_snap.exists:
				cached_tree = doc_snap.to_dict().get('tree')
				
		if cached_tree:
			# Cache Hit
			results.append(cached_tree)
		else:
			# Cache Miss -> Run spaCy with recursive grouping
			tree = analyze_sentence_recursive(sent_text)
			if not tree:
				raise HTTPException(status_code=400, detail="The sentence is cut-off or misses the main verb(ROOT).")
				
			# Save to Global Cache
			if db:
				try:
					db.collection('global_sentence_cache').document(sent_hash).set({
						'text': sent_text,
						'tree': tree,
						'version': '1.0',
						'lang': 'en',
						'createdAt': firestore.SERVER_TIMESTAMP
					})
				except Exception as e:
					print(f"Failed to cache sentence in Firestore: {e}")
					
			results.append(tree)
		
	return results


# Mount the versioned router
app.include_router(api_v1)

# Legacy alias — redirects old clients to v1 without breaking them
@app.post("/analyze", include_in_schema=False)
def analyze_sentence_legacy(request: SentenceRequest):
	"""Deprecated: use /api/v1/analyze instead."""
	return analyze_sentence(request)

if __name__ == "__main__":
	import uvicorn
	uvicorn.run(app, host="127.0.0.1", port=8000)
