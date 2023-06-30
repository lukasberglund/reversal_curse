import os
import diskcache as dc

CACHE_DIR = "cache"


class CacheDummy:
    @staticmethod
    def memoize():
        def decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return decorator


try:
    cache = dc.Cache(os.path.join(CACHE_DIR, "completion_cache"), size_limit=10 * 1e9)
except Exception as e:
    print("Could not create cache. Replacing decorator with dummy. Error:", e)
    cache = CacheDummy()
