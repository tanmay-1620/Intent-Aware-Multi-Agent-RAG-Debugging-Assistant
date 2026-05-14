import os
from langchain_core.documents import Document


def load_code_from_folder(folder_path: str):
    documents = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()

                    # ✅ Only add contextual metadata (NO biasing)
                    enhanced_content = f"""
File Name: {file}
File Path: {file_path}

{content}
"""

                    doc = Document(
                        page_content=enhanced_content,
                        metadata={
                            "source": file_path,
                            "filename": file,
                            "type": "code"
                        }
                    )

                    documents.append(doc)

                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    print(f"💻 Loaded {len(documents)} code files")

    return documents