import ctypes
import os

LLAMA_DLL_PATH = "C:/Users/Ryan/AppData/Local/Programs/Python/Python311/Lib/site-packages/llama_cpp/lib/llama.dll"
assert os.path.exists(LLAMA_DLL_PATH), "⚠️ Llama DLL not found!"

print("🔄 Loading llama.dll...")
ctypes.CDLL(LLAMA_DLL_PATH)
print("✅ Successfully loaded llama.dll!")
