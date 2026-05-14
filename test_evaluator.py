from evaluation.evaluator import score_relevance, score_completeness, score_structure, score_groundedness, final_score

# Test improved relevance for image query
rel = score_relevance('explain this screenshot', '🔍 Visible Text: Aw, Snap! Error code 11 🧠 Meaning: Server connection error 📈 Confidence: Medium', 'image_debug')
print('Image Relevance:', rel)

# Test improved completeness
comp = score_completeness('This is a short answer with some useful information.')
print('Completeness:', comp)

# Test improved structure
struct = score_structure('🔍 Issue: Error code 11 🧠 Meaning: Server error 🛠 Next: Check connection 📈 Confidence: High')
print('Structure:', struct)

# Test groundedness
ground = score_groundedness('Answer text', [], 'image_debug')
print('Groundedness (image):', ground)

# Test final score
final = final_score('explain error', '🔍 Error code 11 🧠 Server issue 🛠 Check DNS 📈 High', [], 'image_debug')
print('Final Scores:', final)