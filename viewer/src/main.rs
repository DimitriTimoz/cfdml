use csv::ReaderBuilder;
use plotters::prelude::*;
use std::collections::HashMap;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Lecture des nœuds
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path("nodes.csv")?;
    let mut nodes: HashMap<String, (f64, f64)> = HashMap::new();

    for result in rdr.records() {
        let record = result?;
        let id = record.get(0).unwrap().to_string();
        let x: f64 = record.get(2).unwrap().parse()?;
        let y: f64 = record.get(3).unwrap().parse()?;
        nodes.insert(id, (x, y));
    }

    // Lecture des arêtes
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path("edges.csv")?;
    let mut edges: Vec<(String, String)> = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let source = record.get(0).unwrap().to_string();
        let target = record.get(1).unwrap().to_string();
        edges.push((source, target));
    }

    // Détermination des limites pour l'échelle
    let xs: Vec<f64> = nodes.values().map(|(x, _)| *x).collect();
    let ys: Vec<f64> = nodes.values().map(|(_, y)| *y).collect();
    let min_x = xs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_x = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_y = ys.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_y = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Création de la zone de dessin
    let root = BitMapBackend::new("graph.png", (1920*2, 1080*2)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .build_cartesian_2d(min_x..max_x, min_y..max_y)?;

    // Désactivation des axes pour un graphe plus propre
    chart.configure_mesh().disable_mesh().draw()?;

    // Dessin des arêtes
    for (source, target) in edges {
        if let (Some(&(x1, y1)), Some(&(x2, y2))) = (nodes.get(&source), nodes.get(&target)) {
            chart.draw_series(LineSeries::new(vec![(x1, y1), (x2, y2)], &BLACK))?;
        }
    }

    // Dessin des nœuds
    for &(x, y) in nodes.values() {
        chart.draw_series(PointSeries::of_element(
            vec![(x, y)],
            2, // taille du point
            &RED,
            &|c, s, st| {
                EmptyElement::at(c) + Circle::new((0, 0), s, st.filled())
            },
        ))?;
    }

    Ok(())
}
