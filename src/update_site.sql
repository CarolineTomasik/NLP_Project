UPDATE articles SET site = SUBSTR(SUBSTR(url, INSTR(url, '//') + 2), 0, INSTR(SUBSTR(url, INSTR(url, '//') + 2), '/')) 
