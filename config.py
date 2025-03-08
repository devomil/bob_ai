CONFIG = {
    'languages': [
        'python',
        'javascript',
        'java',
        'cpp',
        'rust',
        'go'
    ],
    'models': {
        'code_generation': 'microsoft/codebert-base',
        'language_detection': 'microsoft/codebert-mlm',
    },
    'screen': {
        'capture_interval': 0.1,
        'confidence_threshold': 0.8,
        'ocr_lang': 'eng'
    },
    'learning': {
        'repo_limit': 10,
        'analysis_depth': 3,
        'update_interval': 86400
    }
}
