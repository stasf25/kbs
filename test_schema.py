# test_schema.py
import sys, os
sys.path.insert(0, "service/kbs")
os.environ["KBS_JWT_SECRET"] = "dev-test"  # минимальная конфигурация

try:
    from main import app
    print("✅ App imported")
    schema = app.openapi()
    print(f"✅ Schema generated: {len(schema.get('paths', {}))} paths")
except RecursionError as e:
    print(f"❌ RecursionError (циклическая ссылка в моделях): {e}")
    import traceback; traceback.print_exc()
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()
