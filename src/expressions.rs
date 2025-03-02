#![allow(clippy::unused_unit)]
use polars::prelude::*;
use polars_arrow::legacy::prelude::*;
use polars::prelude::arity::binary_elementwise_values;
use pyo3_polars::derive::polars_expr;

// Baseline fuzzy comparison of two strings
fn gram_str(string1: &str, string2: &str) -> f64 {
    let mut s1: Vec<(char, char)> = Vec::new();
    let mut s2: Vec<(char, char)> = Vec::new();
    let mut sunion: Vec<(char, char)> = Vec::new();

    // If either string is empty, return 0.0
    if string1.is_empty() || string2.is_empty() {
        return 0.0;
    }
    // Do standard, less expensive comparison, return 1.0 if both strings are the same
    else if string1 == string2 {
        return 1.0;
    }
    else {
        let l1 = string1.len() - 1;
        s1.reserve(l1 + 2);
        // Add a space at the beginning for capturing transpositions if loose matching
        s1.push((' ', string1.chars().next().unwrap()));
        let chars1: Vec<char> = string1.chars().collect();
        for i in 0..l1 {
            s1.push((chars1[i], chars1[i + 1]));
        }
        // Add a space at the end if loose matching
        s1.push((chars1[l1], ' '));

        let l2 = string2.len() - 1;
        // Repeat for the second string and take the union of both strings
        sunion.reserve(l1 + l2 + 4);
        sunion.extend(&s1);
        s2.reserve(l2 + 2);
        s2.push((' ', string2.chars().next().unwrap()));
        sunion.push((' ', string2.chars().next().unwrap()));
        let chars2: Vec<char> = string2.chars().collect();
        for i in 0..l2 {
            s2.push((chars2[i], chars2[i + 1]));
            sunion.push((chars2[i], chars2[i + 1]));
        }
        s2.push((chars2[l2], ' '));
        sunion.push((chars2[l2], ' '));

        // Sort and deduplicate the union vector
        sunion.sort();
        sunion.dedup();

        // Calculate the frequency at which each unique char pairing occurs in both strings
        let lu = sunion.len();
        let mut f1: Vec<i32> = Vec::with_capacity(lu);
        let mut f2: Vec<i32> = Vec::with_capacity(lu);
        for bi in &sunion {
            f1.push(s1.iter().filter(|&&x| x == *bi).count() as i32);
            f2.push(s2.iter().filter(|&&x| x == *bi).count() as i32);
        }

        // Compute similarity score using dot product of both frequency vectors
        let dot_product: i32 = f1.iter().zip(&f2).map(|(a, b)| a * b).sum();
        let magnitude1: f64 = f1.iter().map(|&x| x * x).sum::<i32>() as f64;
        let magnitude2: f64 = f2.iter().map(|&x| x * x).sum::<i32>() as f64;

        let jacc = (dot_product as f64) / (magnitude1.sqrt() * magnitude2.sqrt());

        jacc
    }
}

#[polars_expr(output_type=Float64)]
fn gram(inputs: &[Series]) -> PolarsResult<Series> {
    let lstr = inputs[0].str()?;
    let rstr = inputs[1].str()?;
    let out: Float64Chunked = binary_elementwise_values(
        lstr,
        rstr,
        |x, y| gram_str(x, y)
    );
    Ok(out.into_series())
}