from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image, PageBreak
# from pylab import *
import streamlit as st
import pandas as pd
from reportlab.lib.pagesizes import letter, landscape
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Table, Image
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import dataframe_image as dfi
from Analyser.loader import get_good_naming
# from Analyser.loader import get_all_logs
from Analyser.phases_analysis import upwind_downwind
from Analyser.phases_analysis import get_phases
from Analyser.phases_analysis import create_dataframe_Boat
from Analyser.phases_analysis import get_phase_report
from Analyser.phases_analysis import get_all_phases
from Analyser.report_plots import get_subplots
from Analyser.report_plots import get_subplots_v2
from Analyser.report_plots import get_x_y_graph
from Analyser.manoeuvres_analysis import get_man_summary
from Analyser.manoeuvres_analysis import get_man_details_v2
from Analyser.manoeuvres_analysis import average_plot
from Analyser.report_plots import get_subplots_man
from Analyser.loader import get_sim_crew
from Analyser.loader import get_metadata, get_one_boat_logs_bis
import seaborn as sns
import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
cm = sns.light_palette("green", as_cmap=True)
colors = ["RdBu"]
cmap = matplotlib.colors.ListedColormap(colors)

naming = pd.DataFrame() # pd.read_csv("Good_notes_240320.csv")
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the full path to the CSV file
csv_path = os.path.join(script_dir, 'name_mapping.csv')
pngs_path = os.path.join(script_dir, 'pngs')
dico = pd.read_csv(csv_path, sep=";")

period = 8
min_bsp = 20
perc_min = 90
perc_max = 110
dev_hdg = 2
dev_bsp = 1.5

man_list = ["Datetime", 'x', 'y', "JibSheetingAngle", "Tgt_MainTraveller", "Tack", 'HDG',
            'TWA', 'BSP', 'VMG', 'JibTrack', 'VMG%', 'FoilPort_FlapAngle', 'FoilStbd_FlapAngle', "Crew"]

timing = 4
percent = 90
bsp_limit = 10

color_list = ["green", "cyan", "blue", "red",
              "yellow", "purple", "pink", "grey", "orange"]

overall_variables = ['VMG', 'VMG%', 'BSP', 'TWA', 'AWA', 'AWS']
general_setup = ['Heel', 'HullAltitude', 'Leeway',
                 'Rudder_Angle']
hydro = ['Trim', 'FoilCant', 'FoilCant_eff', 'Sink',
         'Flap', 'sum_Flap', 'Rudder_Rake', 'sum_Rudder_Rake']
stability = ['MainTraveller', 'sum_MainTraveller',
             'dev_Heel', 'dev_Trim', 'sum_Heel', 'dev_TWA']
aero1 = ['MastRotation', 'MainSheet', 'MainCunninghamLoad_kg', 'Clew',
         'JibSheetLoad_kg', 'JibCunninghamLoad_kg'
         ]
aero2 = ["MainFootCamberAngle", "MainMidCamber", "MainTwist", "JibMidCamber_pc",
         "JibTwist", "MainSheetnoLoad", "JibTrack", "sum_JibSheetLoad_kg"]


variable_list = [overall_variables,
                 general_setup, hydro, stability, aero1, aero2]
variable_dict = {
    'overall_variables': overall_variables,
    'general_setup': general_setup,
    'hydro': hydro,
    'stability': stability,
    'aero1': aero1,
    'aero2': aero2
}

# Get the names of the dictionary keys
new_var_list = list(variable_dict.keys())

correlations_to_plots = [
    ["sum_Rudder_Angle", "sum_MainTraveller"],
    ["sum_MainTraveller", "sum_Heel"],
    ["JibTrack", "VMG%"],
    ["AWA", "VMG%"],
]

var_of_interest = ["VMG%", "BSP", "TWA", "FoilCant",
                   "FoilCant_eff", "Heel", "dev_Heel", "dev_TWA", "HullAltitude"]

overall_variables_man = ['entry_bsp', 'entry_twa', 'exit_bsp',
                         'exit_twa', 'vmg_loss', 'vmg_loss_target', 'distance']
entry_var = ['entry_mainsheet', 'entry_traveller', 'entry_flap', 'entry_jib_sheet',
             'entry_rh', 'entry_heel']
exit_var = ['exit_mainsheet', 'exit_traveller', 'exit_flap', 'exit_jib_sheet',
            'exit_rh', 'exit_heel']

turn_var = ['turn_min_rh', 'max_yaw_rate', 'turn_time',
            'dev_yaw_rate', 'turn_rh', 'turn_heel', 'poptime']
build_var = ['build_bsp', 'build_twa',
             'build_bsp_stab', 'build_flap_sum', 'build_traveller_sum', 'build_rh',
             'build_sheet_sum', 'build_jib_sheet_sum'
             ]
variable_list_man = [overall_variables_man,
                     entry_var, exit_var, turn_var, build_var]
variable_dict_man = {
    'overall_variables_man': overall_variables_man,
    'entry_var': entry_var,
    'exit_var': exit_var,
    'turn_var': turn_var,
    'build_var': build_var,
}

# Get the names of the dictionary keys
new_var_list_man = list(variable_dict_man.keys())


def create_page_with_layout(file_name):
    H = 350
    W = 320
    # Adjust the margins here (in inches)
    left_margin = 0.2  # Adjust as needed
    right_margin = 0.2  # Adjust as needed
    bottom_margin = .1

    doc = SimpleDocTemplate(file_name, pagesize=landscape(letter),
                            leftMargin=left_margin,
                            rightMargin=right_margin,
                            bottomMargin=bottom_margin)
    styles = getSampleStyleSheet()
    # doc = SimpleDocTemplate(file_name, pagesize=landscape(letter))
    # styles = getSampleStyleSheet()

    # List to hold elements
    elements = []

    # Add title
    title_text = "Overview SimSession"
    elements.append(Paragraph(title_text, styles["Title"]))

    # Create table with upwind and downwind plots side by side
    table_data_1 = [
        [
            # [Image(f'{pngs_path}/naming.png', width=220, height=220), Image('pngs/wind_log.png',  width=220, height=200),
            # Image('pngs/', width=100, height=100)
        ],
        [Image(f'{pngs_path}/tracking.png', width=300, height=200), Image(f'{pngs_path}/wind_phases.png',  width=220, height=200),
         Image(f'{pngs_path}/crew_perc_phases.png', width=150, height=150)]
    ]
    table1 = Table(table_data_1)
    elements.append(table1)
    elements.append(PageBreak())

    table_data_2 = [
        [Paragraph("UPWIND", styles["Heading2"]),
         Paragraph("DOWNWIND", styles["Heading2"])],
        #[Image(f'{pngs_path}/report_up.png', width=230, height=360),
        # Image(f'{pngs_path}/report_down.png',  width=230, height=360)],

    ]
    table2 = Table(table_data_2)
    elements.append(table2)

    elements.append(PageBreak())
    i = 0
    for var in list(variable_dict.keys()):
        # Add title
        title_text = f"{var} analysis"
        elements.append(Paragraph(title_text, styles["Title"]))
        if len(upwind) > 10:
            if len(downwind) > 10:
                table_data = [
                    [Paragraph("UPWIND", styles["Heading2"]),
                     Paragraph("DOWNWIND", styles["Heading2"])],
                    [Image(f'{pngs_path}/fig_n{i}.png', width=W, height=H),
                     Image(f'{pngs_path}/fig_n{i+1}.png',  width=W, height=H)]
                ]
                table = Table(table_data)
                elements.append(table)
                elements.append(PageBreak())
                i += 2
            else:
                table_data = [
                    [Paragraph("UPWIND", styles["Heading2"])],
                    [Image(f'{pngs_path}/fig_n{i}.png',
                           width=W*1.5, height=H*1.2)]
                ]
                table = Table(table_data)
                # Center align the content within each cell
                table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
                ]))
                elements.append(table)
                elements.append(PageBreak())
                i += 2

        else:
            table_data = [
                [Paragraph("DOWNWIND", styles["Heading2"])],
                [Image(f'{pngs_path}/fig_n{i+1}.png', width=W*1.5, height=H*1.2)]
            ]
            table = Table(table_data)
            elements.append(table)
            elements.append(PageBreak())
            i += 2

    if len(upwind) > 10:
        title_text = "Correlation Table Upwind"
        elements.append(Paragraph(title_text, styles["Title"]))

        # Create table with upwind and downwind plots side by side
        table_correlation_upwind = [
            [Paragraph("UPWIND", styles["Heading2"])],
            [Image(f'{pngs_path}/fig_n12.png',  width=W, height=H),
             Image(f'{pngs_path}/heatmap_up.png',  width=W, height=H)]
        ]
        table_corr_up = Table(table_correlation_upwind)
        elements.append(table_corr_up)
        elements.append(PageBreak())

    elif len(downwind) > 10:
        title_text = "Correlation Table Downwind"
        elements.append(Paragraph(title_text, styles["Title"]))

        # Create table with upwind and downwind plots side by side
        table_correlation_downwind = [
            [Paragraph("DOWNWIND", styles["Heading2"])],
            [Image(f'{pngs_path}/fig_n13.png',  width=W, height=H),
             Image(f'{pngs_path}/heatmap_down.png',  width=W, height=H)]
        ]
        table_corr_down = Table(table_correlation_downwind)
        elements.append(table_corr_down)
        elements.append(PageBreak())
    else:
        elements.append(PageBreak())

    # table_optimum = [
     #       [Paragraph("Optimum", styles["Heading2"])],
    #        [Image('pngs/fig_optim.png',  width=W, height=H), Image('pngs/fig_optim2.png',  width=W, height=H)]
     #   ]
    # table_optim = Table(table_optimum)
    # elements.append(table_optim)
    # elements.append(PageBreak())
    table_data_3 = [
        [Paragraph("Tacks", styles["Heading2"]),
         Paragraph("Gybes", styles["Heading2"])],
        #[Image(f'{pngs_path}/tack_table.png', width=230, height=360),
        # Image(f'{pngs_path}/gybe_table.png',  width=230, height=360)],
            
    ]
    table3 = Table(table_data_3)
    elements.append(table3)

    elements.append(PageBreak())
    i = 0
    for var in list(variable_dict_man.keys()):
        # Add title
        title_text = f"{var} analysis"
        elements.append(Paragraph(title_text, styles["Title"]))
        if len(man[man.man_type == 'tack']) > 2:
            if len(man[man.man_type == 'gybe']) > 2:
                table_data = [
                    [Paragraph("TACKS", styles["Heading2"]),
                     Paragraph("GYBES", styles["Heading2"])],
                    [Image(f'{pngs_path}/fig_man_n{i}.png', width=W, height=H),
                     Image(f'{pngs_path}/fig_man_n{i+1}.png',  width=W, height=H)]
                ]
                table = Table(table_data)
                elements.append(table)
                elements.append(PageBreak())
                i += 2
            else:
                table_data = [
                    [Paragraph("TACKS", styles["Heading2"])],
                    [Image(f'{pngs_path}/fig_man_n{i}.png',
                           width=W*1.5, height=H*1.2)]
                ]
                table = Table(table_data)
                # Center align the content within each cell
                table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
                ]))
                elements.append(table)
                elements.append(PageBreak())
                i += 2

        else:
            table_data = [
                [Paragraph("GYBES", styles["Heading2"])],
                [Image(f'{pngs_path}/fig_man_n{i+1}.png',
                       width=W*1.5, height=H*1.2)]
            ]
            table = Table(table_data)
            elements.append(table)
            elements.append(PageBreak())
            i += 2

    # Build PDF
    doc.build(elements)
    return file_name
# Create the PDF with the specified layout
# create_page_with_layout(f'{pdf_name}')


# Streamlit app
st.title("Artemis PerfReport Builder for SIM Session")

# Upload CSV file

uploaded_csv = st.file_uploader(
    "Upload CSV file", type="csv", accept_multiple_files=True)
sample = st.text_input(
    "Enter the name of the Pdf report 👇")
# Button to generate PDF Report
if st.button("Generate PDF Report") and uploaded_csv:
    st.write("Loading data...")
    time.sleep(1)

    all_logs = get_one_boat_logs_bis(uploaded_csv, dico, naming, 5)
    # all_logs = pd.read_csv(uploaded_csv)

    all_logs.Datetime = pd.to_datetime(all_logs.Datetime)
    all_logs = all_logs[all_logs.Datetime.isin(all_logs.Datetime.dropna())]
    all_logs = all_logs.sort_values(by='Datetime')
    all_logs = all_logs.drop(columns="New_datetime")
    all_logs = all_logs.reset_index()
    all_logs['Crew'] = ''
    # all_logs.drop(columns='level_0', inplace=True)
    st.write("Creating phases...")
    time.sleep(1)
    all_phases = get_phases(all_logs, period, min_bsp, dev_hdg,
                            dev_bsp, perc_min, perc_max, naming, TWA_ref="TWA")

    st.write(f"Number of phases detected : {len(all_phases)}")

    # Function to create PDF report

    # for race in [value for value in all_logs.run_goal.unique() if value != np.nan and not isinstance(value, float)]:
    # for race in ['Race 1', 'Race 2', 'Race 3', 'Race 4']:#, 'Race7', 'Race2', 'Race4', 'Race5', 'Race8']:
    filtered_logs = all_logs.copy()
    # filtered_logs['csv_file'] = filtered_logs['Boat'] + filtered_logs['run_goal']

    filtered_logs.Datetime = pd.to_datetime(filtered_logs.Datetime)
    filtered_logs.sort_values(by='Datetime', inplace=True)

    plt.figure(figsize=(20, 20))
    rcParams['figure.figsize'] = 22, 22
    sns.scatterplot(data=filtered_logs.reset_index(), x='x', y='y', hue='TACK')
    plt.savefig(f'{pngs_path}/tracking.png', bbox_inches='tight')

    sns.set(rc={"figure.figsize": (8, 4)})

    plt.subplot(1, 2, 1)
    ax = sns.histplot(data=filtered_logs.reset_index(),
                      x='TWS_kts', hue='TACK')

    plt.subplot(1, 2, 2)
    ax = sns.histplot(data=filtered_logs.reset_index(), x='TWD', hue='TACK')

    plt.savefig(f"{pngs_path}/wind_log.png")
    plt.show()
    # Filters used (standard would be 18, 2, 2)

    # filtered_logs = filtered_logs.set_index(filtered_logs.Datetime)
    # filtered_logs.sort_index(inplace=True)
    # get_all_phases(filtered_logs, period, min_bsp, dev_hdg, dev_bsp, perc_min, perc_max, naming ,TWA_ref="TWA")
    all_phases = all_phases.copy()
    st.write("Creating manoeuvres...")
    time.sleep(1)
    man, all_man = get_man_details_v2(
        filtered_logs, timing, percent, bsp_limit, man_list)
    st.write(f"Number of manoeuvres :{len(man)}")
    time.sleep(1)
    upwind, downwind = upwind_downwind(all_phases)
    # .style.background_gradient(cmap="YlGnBu", axis=0).set_precision(2)
    styled_df = (all_phases.groupby('TACK').count()[
        ['Time']]/all_phases.Time.count()*100).round(1)
    styled_df.reset_index()
    styled_df = styled_df.rename(columns={'Time': 'Percentage'})
    styled_df['Perc of sailing'] = (all_phases.groupby('TACK').count(
    )[['Time']]*period*5/filtered_logs.groupby('TACK')[['Time']].count()*100).round(1)
    plt.figure(figsize=(5, 5))
    plt.table(cellText=styled_df.reset_index().values,
              colLabels=styled_df.reset_index().columns, loc='center')
    plt.axis('off')  # Hide axis

    # Save the plot as a PNG image
    # bbox_inches='tight' ensures no extra whitespace
    plt.savefig(f'{pngs_path}/boat_perc_phases.png', bbox_inches='tight')
    st.write("Creating the pngs...")
    time.sleep(1)
    # Show the plot (optional)
    plt.show()
    # .style.background_gradient(cmap="YlGnBu", axis=0).set_precision(2)
    styled_df = (all_phases.groupby(['TACK']).count()[
        ['Time']]/all_phases.Time.count()*100).round(1)
    styled_df.reset_index()

    plt.figure(figsize=(5, 5))
    plt.table(cellText=styled_df.reset_index().values,
              colLabels=styled_df.reset_index().columns, loc='center')
    plt.axis('off')  # Hide axis

    # Save the plot as a PNG image
    # bbox_inches='tight' ensures no extra whitespace
    plt.savefig(f'{pngs_path}/crew_perc_phases.png', bbox_inches='tight')

    # Show the plot (optional)
    # plt.show()

    sns.set(rc={"figure.figsize": (8, 4)})

    plt.subplot(1, 2, 1)
    ax = sns.histplot(data=all_phases, x='TWS_kts', hue='TACK')

    plt.subplot(1, 2, 2)
    ax = sns.histplot(data=all_phases, x='TWD', hue='TACK')

    plt.savefig(f"{pngs_path}/wind_phases.png")
    # plt.show()
    plt.close()
    plt.figure(figsize=(10, 10))
    rcParams['figure.figsize'] = 10, 10
    k = 0
    for var_list in variable_list:
        i = 0
        for df in upwind, downwind:
            if len(df) < 10:
                k += 1
                i = 1
            else:
                if i == 1:
                    rcParams['figure.figsize'] = 10, 10
                    get_subplots(df, var_list, "TWS_kts", "TACK",
                                 1, color_list, dico, f'{pngs_path}/fig_n{k}')
                    k += 1
                else:
                    rcParams['figure.figsize'] = 10, 10
                    get_subplots(df, var_list, "TWS_kts", "TACK",
                                 1, color_list, dico, f'{pngs_path}/fig_n{k}')
                    k += 1
    plt.close()

    for df in upwind, downwind:
        if len(df) < 10:
            k += 1
        else:
            get_subplots_v2(df, correlations_to_plots,  "TACK",
                            1, color_list, dico, f'{pngs_path}/fig_n{k}')
            k += 1

    k = 0
    for var_list in variable_list_man:
        i = 0
        for mantype in ['tack', 'gybe']:
            df = man[man.man_type == mantype]
            # if len(df)<2:
            #  k+=1
            #  i=1
            # else:
            # if i==1:
            #        rcParams['figure.figsize'] = 10,10
            #        get_subplots_man(df, var_list, "tws", "crew", 1, color_list ,dico, f'pngs/fig_man_n{k}')
            #        k+=1
            # else :
            rcParams['figure.figsize'] = 10, 10
            get_subplots_man(df, var_list, "tws", "tackside",
                             1, color_list, dico, f'{pngs_path}/fig_man_n{k}')
            k += 1

    sns.heatmap(downwind[var_of_interest].corr(), annot=True)
    plt.savefig(f'{pngs_path}/heatmap_down.png')
    plt.close()

    # report_up = get_phase_report(upwind)
    # dfi.export(report_up, f'{pngs_path}/report_up.png', table_conversion = 'matplotlib')
    # plt.close()
    # report_down = get_phase_report(downwind)
    # dfi.export(report_down, f'{pngs_path}/report_down.png', table_conversion = 'matplotlib')
    # plt.close()

    sns.heatmap(upwind[var_of_interest].corr(), annot=True)
    plt.savefig(f'{pngs_path}/heatmap_up.png')
    plt.close()
    #tack_table = man[man.man_type == 'tack'].groupby('tackside').median()[['tws', 'flying', 'vmg_loss', 'vmg_loss_target', 'distance', 'entry_bsp', 'exit_bsp',
    ##                                                                       'min_bsp', 'entry_twa', 'exit_twa', 'exit_vmg',
    #                                                                       'entry_mainsheet', 'entry_traveller', 'entry_jib_sheet',
    #                                                                       'entry_rh_stability', 'entry_rh', 'entry_heel', 'entry_pitch',
    #                                                                       'exit_traveller', 'exit_mainsheet', 'exit_jib_sheet',
    #                                                                       'exit_rh', 'exit_heel', 'exit_pitch', 'max_yaw_rate',
     #                                                                      'turn_min_rh', 'turn_time', 'dev_yaw_rate',
     #                                                                      'turn_rh', 'turn_pitch', 'turn_heel', 'poptime']].T.style.format(precision=1).background_gradient(cmap="YlGnBu", axis=1)
    # dfi.export(tack_table, f'{pngs_path}/tack_table.png',table_conversion = 'matplotlib')
    #plt.close()
    #gybe_table = man[man.man_type == 'gybe'].groupby('tackside').median()[['tws', 'flying', 'vmg_loss', 'vmg_loss_target', 'distance', 'entry_bsp', 'exit_bsp',
    #                                                                       'min_bsp', 'entry_twa', 'exit_twa', 'exit_vmg',
    #                                                                       'entry_mainsheet', 'entry_traveller', 'entry_jib_sheet',
     #                                                                      'entry_rh_stability', 'entry_rh', 'entry_heel', 'entry_pitch',
    #                                                                       'exit_traveller', 'exit_mainsheet', 'exit_jib_sheet',
     #                                                                      'exit_rh', 'exit_heel', 'exit_pitch', 'max_yaw_rate',
     #                                                                      'turn_min_rh', 'turn_time', 'dev_yaw_rate',
     #                                                                      'turn_rh', 'turn_pitch', 'turn_heel', 'poptime']].T.style.format(precision=1).background_gradient(cmap="YlGnBu", axis=1)
    # dfi.export(gybe_table, f'{pngs_path}/gybe_table.png',table_conversion = 'matplotlib')
    #plt.close()

    # pdf_name = f"pdfs/PerfReport_{race}_240319"
    pdf_file_path = f"{sample}.pdf"
    st.write("Generate PDF...")
    time.sleep(1)
    file_path = create_page_with_layout(pdf_file_path)

    with open(file_path, "rb") as file:
        pdf_data = file.read()
    st.success('PDF generation completed!')
    # Provide a download button in the Streamlit app
    st.download_button(
        label="Download PDF",
        data=pdf_data,
        file_name=f"{sample}.pdf",
        mime="application/pdf"
    )
