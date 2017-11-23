import os
from tabula import read_pdf
import PyPDF2 as pdf
import json

REPLACE_CHARACTERS = ['\n', '\t']

def get_page_key(page_num, ls_file_contents):
	'''
	@description		: obtain the keyword from the page; it's considered to be a part of the header of the page and starting with *

	@ls_file_contents	: pypdf2 object that has contents of the pdf
	@returns			: returns a string consisting of the keyword name. If nothing was found, an empty string is returned.
	'''
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


def get_pdf_dict(key_file_path, page_numbers, only_page_num_dict=False):
	'''
	@description		: returns a dictionary of keywords and corresponding tables and page numbers in lists

	@page_numbers		: list of page numbers for which keys and tables must be extracted; if "All" is given, all pages are analyzed
	@only_page_num_dict	: (boolean; default:False) if this is set to True, only page numbers are written out for given key word manual
	@returns			: returns a dictionary of keywords and corresponding tables and page numbers in lists
	'''
	
	ls_key_dict = {}
	ls_file = open(key_file_path, 'rb')
	ls_file_contents = pdf.PdfFileReader(ls_file)

	if isinstance(page_numbers, int):
		page_numbers = [page_numbers]
	
	if page_numbers == "All":
		total_pages = ls_file_contents.getNumPages()
		page_numbers = list(range(1, total_pages+1))

	for page_num in page_numbers:
		page_key = get_page_key(page_num, ls_file_contents)
		if page_key == "":
			continue

		if ls_key_dict.get(page_key):
			ls_key_dict[page_key]['page_numbers'].append(page_num)
		else:
			ls_key_dict[page_key] = {'tables' : [], 'page_numbers' : [page_num]}
		
		if only_page_num_dict:
			continue
		
		#page nos. for this function start from 1
		page_tables = read_pdf(key_file_path, pages=[page_num], multiple_tables=True)
		if not page_tables:
			ls_key_dict[page_key]['page_numbers'].pop(-1)
			continue

		ls_key_dict[page_key]['tables'].append(page_tables)

	return ls_key_dict


def main():
	key_file_path = "res/LSDyna_Manual_1.pdf"
	page_numbers = [1,2,3,4,5,6]

	ls_key_dict = get_pdf_dict(key_file_path, "All", True)
	#print(ls_key_dict)

	with open('res/keyword_dict.txt', 'w') as outfile:
		json.dump(ls_key_dict, outfile)


main()
