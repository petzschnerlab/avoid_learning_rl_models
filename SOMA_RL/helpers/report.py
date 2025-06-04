import pandas as pd
import dataframe_image as dfi


class Report:

    """
    Class to hold reporting functions for the SOMA project
    """
    
    def table_to_png(self, table: pd.DataFrame, save_name: str = "SOMA_AL/plots/tables/Table.png") -> None:

        """
        Converts a table to a png

        Parameters
        ----------
        table : pd.DataFrame
            The table to convert
        save_name : str
            The name to save the table as

        Returns (External)
        ------------------
        Image: PNG
            The table as a png
        """
        
        #Format titles as titles
        for i in range(len(table)):
            table.index.values[i] = table.index.values[i].title()  
        table.columns = table.columns.str.title()
        table.columns.name = None   

        #Format the table
        table = table.style.set_table_styles([{'selector': 'th', 'props': [('font-size', '10pt'), 
                                                                           ('text-align', 'center'), 
                                                                           ('background-color', '#FFFFFF')]},

                                                {'selector': 'td', 'props': [('font-size', '10pt'), 
                                                                             ('text-align', 'center'), 
                                                                             ('background-color', '#FFFFFF')]},

                                                {'selector': '', 'props': [('border-top', '1px solid black'), 
                                                                           ('border-bottom', '1px solid black'),
                                                                           ('border-left', '1px solid white'),
                                                                           ('border-right', '1px solid white')]},])
        
        #Save the table as a png
        dfi.export(table, save_name, table_conversion="selenium", max_rows=-1)