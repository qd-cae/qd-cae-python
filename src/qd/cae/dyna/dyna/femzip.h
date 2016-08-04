

#ifdef FEMZIP_LCASE_USCORE
#define GEOMETRY_READ geometry_read_
#define FEMUNZIPLIB_VERSION femunziplib_version_
#define FEMUNZIPLIB_VERSION_FILE femunziplib_version_file
#define STATES_READ states_read_
#define GLOBAL_READ global_read_
#define DISP_READ_ONLY disp_read_only_
#define STATES_CLOSE states_close_
#define CTIMES_READ ctimes_read_
#define RD_RECORD rd_record_
#define SET_RECORD set_record_
#define SET_LINERD set_linerd_
#define CT_READG ct_readg_
#define CT_READ ct_read_
#define FREE_ALL free_all_
#define ACTIVITY_READ activity_read_
#define DISP_READ disp_read_
#define STATES_READ states_read_
#define POST_READ post_read_
#define POST_READ_MOD post_read_mod_
#define POST_CLOSE post_close_
#define OPEN_READ open_read_
#define CLOSE_READ close_read_
#define READ_VAR read_var_
#define TIMES_SET_ZERO times_set_zero_
#define TIMES_GLUE_BEGIN times_glue_begin_
#define TIMES_GLUE_END times_glue_end_
#define TIMES_PRINT_ANALYSIS times_print_analysis_
#define UNCOMPRESS_RESETUP uncompress_resetup_
#define UNCOMPRESS_RESETUP_CLEAN uncompress_resetup_clean_
#define GET_SIZE get_Size
#define GET_TITLE get_Title
#define GET_TITLE_OLD get_Title_old
#define PART_TITLES_READ part_titles_read_
#else
#define GEOMETRY_READ geometry_read
#define FEMUNZIPLIB_VERSION femunziplib_version
#define FEMUNZIPLIB_VERSION_FILE femunziplib_version_file
#define STATES_READ states_read
#define GLOBAL_READ global_read
#define DISP_READ_ONLY disp_read_only
#define STATES_CLOSE states_close
#define CTIMES_READ ctimes_read
#define RD_RECORD rd_record
#define SET_RECORD set_record
#define SET_LINERD set_linerd
#define CT_READG ct_readg
#define CT_READ ct_read
#define FREE_ALL free_all
#define ACTIVITY_READ activity_read
#define DISP_READ disp_read
#define STATES_READ states_read
#define POST_READ post_read
#define POST_READ_MOD post_read_mod
#define POST_CLOSE post_close
#define OPEN_READ open_read
#define CLOSE_READ close_read
#define READ_VAR read_var
#define UNCOMPRESS_RESETUP uncompress_resetup
#define UNCOMPRESS_RESETUP_CLEAN uncompress_resetup_clean
#define TIMES_SET_ZERO times_set_zero
#define TIMES_PRINT_ANALYSIS times_print_analysis
#define TIMES_GLUE_BEGIN times_glue_begin
#define TIMES_GLUE_END times_glue_end
#define GET_SIZE get_Size
#define GET_TITLE get_Title
#define GET_TITLE_OLD get_Title_old
#define PART_TITLES_READ part_titles_read
#endif


typedef struct
{
int number_of_variables;
int global_var;
int global_type;
int global_number;
int part_var;
int part_type;
int part_number;
int nodal_var;
int nodal_type;
int nodal_number;
int inodal;
int cfd_var;
int cfd_type;
int cfd_number;
int thick_var;
int thick_type;
int thick_number;
int shell_var;
int shell_type;
int shell_number;
int solid_var;
int solid_type;
int solid_number;
int D1_var;
int D1_type;
int D1_number;
int tool_var;
int tool_type;
int tool_number;
int FPM_var;
int FPM_type;
int FPM_number;
int SPH_var;
int SPH_type;
int SPH_number;
int CPM_GEOM_var;
int CPM_GEOM_type;
int CPM_GEOM_number;
int CPM_INT_var;
int CPM_INT_type;
int CPM_INT_number;
int CPM_var;
int CPM_type;
int CPM_number;
int RADIOSS_special_var;
int RADIOSS_special_type;
int RADIOSS_special_number;
int geometry_var;
int geometry_type;
int geometry_number;
int number_of_nodes;
int number_of_solid_elements;
int number_of_thick_shell_elements;
int number_of_1D_elements;
int number_of_tool_elements;
int number_of_shell_elements;
int number_of_solid_element_neighbors;
int number_of_timesteps;
}
ind;


typedef struct
{
char title[33];
}
TitleS;

extern void GEOMETRY_READ(int*, int*, int*, int*, int*, int*, int*, int*);

/* extern void TIMES_READ(int*, int*, float*,int*); */
extern void CTIMES_READ(int*, int*, int*, float*, int*);
extern void ACTIVITY_READ(int*, int*, int*, int*, int*);
extern void DISP_READ(int*, int*, int*, int*, int*);
extern void DISP_READ_ONLY(int*, int*, int*, int*, int*);
extern void GLOBAL_READ(int*, int*, int*, int*, int*);
extern void STATES_READ(int*, int*, int*,int*, int*);
extern void POST_READ(int*, int*, int*, int*, int*);
extern void STATES_CLOSE(int*, int*, int*,int*);
extern void POST_CLOSE(int*, int*, int*,int*);
extern void RD_RECORD(int*, int*);
extern void SET_RECORD(int*, int*);
extern void SET_LINERD(int*, int*, int*);
extern void OPEN_READ(int*,int*,int*,int*,int*);
extern void CLOSE_READ(int*);
extern void wrapinput(int argc, char **argv, int *p1, int *p2, int *l1, int *l2);
extern void UNCOMPRESS_RESETUP(int*);
extern void UNCOMPRESS_RESETUP_CLEAN(int*);
extern void CT_READG(int*, int*);
extern void CT_READ(int*, int*);
extern void FREE_ALL(int*, int*);
extern void POST_READ_MOD(int*, int*, int*, float*, int*, int*, int*, int*);
extern void READ_VAR(float*, int*, int*, int*, int*, int*, int*, int*, int*);
extern void TIMES_SET_ZERO();
extern void TIMES_PRINT_ANALYSIS( char*);
extern void FEMUNZIPLIB_VERSION( float*);
extern void GET_SIZE(char*,int,int,int*,int*,int*,int*,int*,int*);
extern int GET_TITLE(char*, TitleS **, ind *, int*, int*);
extern int GET_TITLE_OLD(char*, TitleS **, ind *, int*, int*);
extern void FEMUNZIPLIB_VERSION_FILE( char*, int*, float*, int*);
extern void PART_TITLES_READ(int*,int*,int*,int*);
