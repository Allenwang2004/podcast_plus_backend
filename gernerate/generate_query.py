# import pandas as pd
import pandas as pd

topics = pd.read_csv('topics.csv')['topic']
unique_topics = topics.drop_duplicates().reset_index(drop=True)

front = "Generate a podcast content between two people discussing "

with open('queries.csv', 'w') as f:
    f.write('topic,query\n')
    for topic in unique_topics:
        query = front + topic
        f.write(f'"{topic}","{query}"\n')