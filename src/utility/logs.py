from datetime import datetime, timedelta

def log(log_str: str, length=30):
    separator = '-' * length
    title_line = log_str.center(length, '-')
    print(f'{separator}\n{title_line}\n{separator}')
    
def log_time(start: datetime, end: datetime, task: str):
    duration: timedelta = end - start
    
    hours, remainder = divmod(duration.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"Task '{task}' started at {start} and ended at {end}. Duration: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds.")