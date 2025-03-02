#![allow(clippy::unused_unit)]
use polars::prelude::*;
use polars_arrow::legacy::prelude::*;
use polars::prelude::arity::binary_elementwise_values;
use pyo3_polars::derive::polars_expr;

// Baseline fuzzy comparison of two strings
fn ratio_str(string1: &str, string2: &str) -> f64 {
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

// Modified fuzzy comparison that compares short string against rolling window of long string
fn partial_ratio_str(string1: &str, string2: &str) -> f64 {
    let mut ls = Vec::new();
    let mut ss = Vec::new();
    let mut sunion = Vec::new();

    if string1.is_empty() || string2.is_empty() {
        return 0.0;
    } else if string1 == string2 {
        return 1.0;
    } else {
        // Compute length of both strings for determining short/long
        let l1 = string1.len() - 1;
        let l2 = string2.len() - 1;
        let mut jacc_max = 0.0;

        // If length is the same do a standard comparison
        if l1 == l2 {
            ls.reserve(l1 + 2);
            ls.push((' ', string1.chars().nth(0).unwrap()));
            for i in 0..l1 {
                ls.push((string1.chars().nth(i).unwrap(), string1.chars().nth(i + 1).unwrap()));
            }
            ls.push((string1.chars().nth(l1).unwrap(), ' '));

            sunion.reserve(l1 + l2 + 4);
            sunion = ls.clone();
            ss.reserve(l2 + 2);
            ss.push((' ', string2.chars().nth(0).unwrap()));
            sunion.push((' ', string2.chars().nth(0).unwrap()));
            for i in 0..l2 {
                ss.push((string2.chars().nth(i).unwrap(), string2.chars().nth(i + 1).unwrap()));
                sunion.push((string2.chars().nth(i).unwrap(), string2.chars().nth(i + 1).unwrap()));
            }
            ss.push((string2.chars().nth(l2).unwrap(), ' '));
            sunion.push((string2.chars().nth(l2).unwrap(), ' '));

            sunion.sort();
            sunion.dedup();

            let lu = sunion.len();
            let mut f1: Vec<i32> = Vec::with_capacity(lu);
            let mut f2: Vec<i32> = Vec::with_capacity(lu);
            for bi in &sunion {
                f1.push(ls.iter().filter(|&x| x == bi).count() as i32);
                f2.push(ss.iter().filter(|&x| x == bi).count() as i32);
            }

            let dot_product: i32 = f1.iter().zip(&f2).map(|(a, b)| a * b).sum();
            let magnitude1: f64 = f1.iter().map(|&x| x * x).sum::<i32>() as f64;
            let magnitude2: f64 = f2.iter().map(|&x| x * x).sum::<i32>() as f64;

            jacc_max = (dot_product as f64) / (magnitude1.sqrt() * magnitude2.sqrt());
        }
        // If string 1 larger than string 2, compare rolling window of string 1 equivalent to size of string 2
        else if l1 > l2 {
            ss.reserve(l2 + 2);
            ss.push((' ', string2.chars().nth(0).unwrap()));
            for i in 0..l2 {
                ss.push((string2.chars().nth(i).unwrap(), string2.chars().nth(i + 1).unwrap()));
            }
            ss.push((string2.chars().nth(l2).unwrap(), ' '));

            // Start from beginning of larger string and roll across until end of window reaches last char
            for d in 0..(l1 - l2 + 1) {
                sunion.reserve(l2 * 2 + 4);
                sunion = ss.clone();
                ls.reserve(l2 + 2);
                ls.push((' ', string1.chars().nth(d).unwrap()));
                sunion.push((' ', string1.chars().nth(d).unwrap()));
                for i in d..(l2 + d) {
                    ls.push((string1.chars().nth(i).unwrap(), string1.chars().nth(i + 1).unwrap()));
                    sunion.push((string1.chars().nth(i).unwrap(), string1.chars().nth(i + 1).unwrap()));
                }
                ls.push((string1.chars().nth(l2 + d).unwrap(), ' '));
                sunion.push((string1.chars().nth(l2 + d).unwrap(), ' '));

                sunion.sort();
                sunion.dedup();

                let lu = sunion.len();
                let mut f1: Vec<i32> = Vec::with_capacity(lu);
                let mut f2: Vec<i32> = Vec::with_capacity(lu);
                for bi in &sunion {
                    f1.push(ls.iter().filter(|&x| x == bi).count() as i32);
                    f2.push(ss.iter().filter(|&x| x == bi).count() as i32);
                }

                let dot_product: i32 = f1.iter().zip(&f2).map(|(a, b)| a * b).sum();
                let magnitude1: f64 = f1.iter().map(|&x| x * x).sum::<i32>() as f64;
                let magnitude2: f64 = f2.iter().map(|&x| x * x).sum::<i32>() as f64;

                let jacc = (dot_product as f64) / (magnitude1.sqrt() * magnitude2.sqrt());

                // If similarity score of 1.0 is achieved, end loop and return score as it can't be improved
                if jacc == 1.0 {
                    jacc_max = jacc;
                    break;
                }
                // Otherwise, if score is greater than current max score, replace
                else if jacc > jacc_max {
                    jacc_max = jacc;
                }

                ls.clear();
                sunion.clear();
            }
        }
        // If string 1 smaller than string 2, compare rolling window of string 2 equivalent to size of string 1
        else {
            ss.reserve(l1 + 2);
            ss.push((' ', string1.chars().nth(0).unwrap()));
            for i in 0..l1 {
                ss.push((string1.chars().nth(i).unwrap(), string1.chars().nth(i + 1).unwrap()));
            }
            ss.push((string1.chars().nth(l1).unwrap(), ' '));

            for d in 0..(l2 - l1 + 1) {
                sunion.reserve(l1 * 2 + 4);
                sunion = ss.clone();
                ls.reserve(l1 + 2);
                ls.push((' ', string2.chars().nth(d).unwrap()));
                sunion.push((' ', string2.chars().nth(d).unwrap()));
                for i in d..(l1 + d) {
                    ls.push((string2.chars().nth(i).unwrap(), string2.chars().nth(i + 1).unwrap()));
                    sunion.push((string2.chars().nth(i).unwrap(), string2.chars().nth(i + 1).unwrap()));
                }
                ls.push((string2.chars().nth(l1 + d).unwrap(), ' '));
                sunion.push((string2.chars().nth(l1 + d).unwrap(), ' '));

                sunion.sort();
                sunion.dedup();

                let lu = sunion.len();
                let mut f1: Vec<i32> = Vec::with_capacity(lu);
                let mut f2: Vec<i32> = Vec::with_capacity(lu);
                for bi in &sunion {
                    f1.push(ss.iter().filter(|&x| x == bi).count() as i32);
                    f2.push(ls.iter().filter(|&x| x == bi).count() as i32);
                }

                let dot_product: i32 = f1.iter().zip(&f2).map(|(a, b)| a * b).sum();
                let magnitude1: f64 = f1.iter().map(|&x| x * x).sum::<i32>() as f64;
                let magnitude2: f64 = f2.iter().map(|&x| x * x).sum::<i32>() as f64;

                let jacc = (dot_product as f64) / (magnitude1.sqrt() * magnitude2.sqrt());

                if jacc == 1.0 {
                    jacc_max = jacc;
                    break;
                } else if jacc > jacc_max {
                    jacc_max = jacc;
                }

                ls.clear();
                sunion.clear();
            }
        }

        jacc_max
    }
}

#[polars_expr(output_type=Float64)]
fn ratio(inputs: &[Series]) -> PolarsResult<Series> {
    let lstr = inputs[0].str()?;
    let rstr = inputs[1].str()?;
    let out: Float64Chunked = binary_elementwise_values(
        lstr,
        rstr,
        |x, y| ratio_str(x, y)
    );
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn partial_ratio(inputs: &[Series]) -> PolarsResult<Series> {
    let lstr = inputs[0].str()?;
    let rstr = inputs[1].str()?;
    let out: Float64Chunked = binary_elementwise_values(
        lstr,
        rstr,
        |x, y| partial_ratio_str(x, y)
    );
    Ok(out.into_series())
}

