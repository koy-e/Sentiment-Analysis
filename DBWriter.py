import pandas as pd
from sqlalchemy import create_engine
import sys
import io

server = 'localhost'
database = 'amazon_reviews'
username = 'sa'
password = 'anextremelystrongpassword4710'
table_name = 'reviews'
engine = create_engine(f"mssql+pymssql://{username}:{password}@{server}/{database}")

# # File paths
# file_path = 'db.tsv'

file_path = '../database/amazon_reviews_us_Apparel_v1_00.tsv'
log_file_path = 'bad_lines.log'


relevant_columns = [
    'review_id', 'product_parent', 'product_title', 'star_rating',
    'helpful_votes', 'verified_purchase', 'review_headline', 'review_body'
]
relevant_columns_indexes = [2, 4, 5, 7, 8, 11, 12, 13]

def read_first(file_path):
    with open(file_path, 'r') as f:
        line = f.readline().split('\t')
    indexes = []
    for i in range (len(line)):
        if(line[i] in relevant_columns):
            indexes.append(i)

    return indexes

def process_line(line):
    # Split the line using the tab ('\t') delimiter
    columns = line.strip().split('\t')
    
    # Specify the expected number of columns (adjust as needed)
    expected_columns = 15

    # Check if the number of columns matches the expected number
    if len(columns) == expected_columns:
        new_columns = []
        for i in range(len(columns)):
            if i in relevant_columns_indexes:
                if(columns[i].isnumeric()):
                    columns[i] = int(columns[i])
                new_columns.append(columns[i])

        return new_columns
    else:
        # Log the line to a file if the number of columns is unexpected
        with open('bad_lines.log', 'a') as log_file:
            log_file.write(f"Unexpected number of columns ({len(columns)}) in line: {line}\n")
        return None

def read_file(file_path, lines_to_skip, lines_to_read, batch_size):
    valid_lines = []
    with open(file_path, 'r') as file:
        next(file)
        for i in range(lines_to_skip):
            next(file)

        for line_count, line in enumerate(file, start=1):
            processed_line = process_line(line)
            if processed_line is not None:
                valid_lines.append(processed_line)

                if len(valid_lines) == batch_size:

                    addToDB(valid_lines)
                    # print(valid_lines)
                    valid_lines = []

            # Check if the specified number of lines has been read
            if line_count >= lines_to_read:
                break

        if valid_lines:
            addToDB(valid_lines)
            #print(valid_lines)

    engine.dispose()

def addToDB(valid_lines):
    df = pd.DataFrame(valid_lines, columns=relevant_columns)
    #could be changed to append
    df.to_sql(table_name, engine, index=False, if_exists='append', schema="dbo")


def runQuery(query):
    df = pd.read_sql_query(query, engine)
    print(df["star_rating"]) 

# read_file(file_path, 0, 10000, 100)


runQuery("SELECT * FROM dbo.reviews WHERE review_id = 'R100BNMAYCTLS';")
# clear_db()