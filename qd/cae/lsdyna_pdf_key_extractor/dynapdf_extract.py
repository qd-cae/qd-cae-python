import os
from tabula import read_pdf
import PyPDF2 as pdf

REPLACE_CHARACTERS = ['\n', '\t']

def get_page_key(page_num, ls_file_contents):
	#get only the headers of the page - most likely the first two words in the page_contents list; also page nos. for this function start from 0
	page_contents = ls_file_contents.getPage(page_num-1).extractText().split(' ')[0:2]
	if not page_contents:
		return ""

	page_key = ""
	#isolate the correct key
	for text in page_contents:
		if "*" in text:
			#considering two cases - [*AIRBAG, *AIRBAG] and [*AIRBAG, *AIRBAG_SIMPLE_PRESSURE_VOLUME]
			if text != page_key and len(text) > len(page_key):
				for rchar in REPLACE_CHARACTERS:
					text = text.replace(rchar, '')

				page_key = text

	return page_key


def get_pdf_dict(key_file_path, page_numbers):
	ls_key_dict = {}
	ls_file = open(key_file_path, 'rb')
	ls_file_contents = pdf.PdfFileReader(ls_file)

	if isinstance(page_numbers, int):
		page_numbers = [page_numbers]

	for page_num in page_numbers:
		page_key = get_page_key(page_num, ls_file_contents)
		if page_key == "":
			continue

		#page nos. for this function start from 1
		page_tables = read_pdf(key_file_path, pages=[page_num], multiple_tables=True)
		if not page_tables:
			continue

		if ls_key_dict.get(page_key):
			ls_key_dict[page_key]['tables'].append(page_tables)
			ls_key_dict[page_key]['page_number'].append(page_num)
		else:
			ls_key_dict[page_key] = {'tables' : [page_tables], 'page_number' : [page_num]}

	return ls_key_dict


def main():
	key_file_path = "res/test02.pdf"
	page_numbers = [1,2,3,4,5,6]

	ls_key_dict = get_pdf_dict(key_file_path, page_numbers)
	print(ls_key_dict)

main()
