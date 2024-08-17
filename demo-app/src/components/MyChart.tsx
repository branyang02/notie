import { BarChart } from "@mui/x-charts/BarChart";

const MyChart = ({ darkMode }: { darkMode: boolean }) => {
    const color = darkMode ? "white" : "black";

    return (
        <BarChart
            series={[
                { data: [35, 44, 24, 34] },
                { data: [51, 6, 49, 30] },
                { data: [15, 25, 30, 50] },
                { data: [60, 50, 15, 25] },
            ]}
            height={290}
            xAxis={[
                {
                    data: ["Q1", "Q2", "Q3", "Q4"],
                    scaleType: "band",
                    tickLabelStyle: {
                        fill: color,
                    },
                },
            ]}
            yAxis={[
                {
                    tickLabelStyle: {
                        fill: color,
                    },
                },
            ]}
            margin={{ top: 10, bottom: 30, left: 40, right: 10 }}
            sx={
                darkMode
                    ? {
                          color: "white",
                          "& .MuiChartsAxis-line": {
                              stroke: "white",
                          },
                          "& .MuiChartsAxis-tick": {
                              stroke: "white",
                          },
                      }
                    : undefined
            }
        />
    );
};

export default MyChart;
