#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use std::f32::consts::TAU;
use rand::prelude::*;

use eframe::{egui::{self, Ui, Sense, Id}, epaint::{Color32, Pos2, Vec2, Stroke, Rect}, emath::NumExt};
use gravity::Simulation;
use gravity;

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1024.0, 768.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Gravity App",
        options,
        Box::new(|_| { Box::<GravityApp>::default() }),
    )
}

#[derive(PartialEq)]
enum Mode {
    Add,
    Remove,
}

// Transforms world cordinates to screen cordinates (and back)
struct FromWorld {
    offset: Vec2,
    scale: f32,
}

impl FromWorld {
    fn transform(&self, position: &gravity::Vec2<f32>) -> Pos2 {
        Pos2::new(self.scale * position.x, self.scale * position.y) + self.offset
    }
    
    fn inverse(&self, position: Pos2) -> gravity::Vec2<f32> {
        let p = position - self.offset;
        gravity::Vec2 { x: p.x / self.scale, y: p.y / self.scale }
    }
    fn inversef(&self, f: f32) -> f32 {
        f / self.scale
    }

    fn pan(&mut self, drag_delta: Vec2) {
        self.offset += drag_delta;
    }

    fn zoom(&mut self, zoom_delta: f32, hover_pos: Pos2) {
        let old_scale = self.scale;
        self.scale *= zoom_delta.sqrt();
        self.offset = hover_pos - self.scale / old_scale * (hover_pos - self.offset);
    }
}

struct Play {
    on: bool,
}
impl Play {
    fn new() -> Play {
        Play { on: false }
    }
}

struct GravityApp {
    rng: rand::rngs::ThreadRng,
    simulation: Simulation,
    // display parameters
    from_world: FromWorld,
    // play parameters
    play: Play,
    // mode
    mode: Mode,
    radius: f32,
    // create mode parameters
    orbital_velocity: bool,
}

impl Default for GravityApp {
    fn default() -> Self {
        Self {
            rng: rand::thread_rng(),
            simulation: Simulation::new(),
            from_world: FromWorld { offset: Vec2::ZERO, scale: 10.0 },
            play: Play::new(),
            mode: Mode::Add,
            radius: 64.0,
            orbital_velocity: true,
        }
    }
}

fn draw(ui: &mut Ui, simulation: &Simulation, spacing: Vec2, from_world: &FromWorld) {
    let size = ui.available_size();
    
    // draw grid
    let color = Color32::from_rgb(00, 0x8b, 0x8b);
    let stroke = Stroke { width: 1.0, color};
    let grid_size = size / spacing;
    let (grid_width, grid_height) = (grid_size.x as usize, grid_size.y as usize);
    let painter = ui.painter();
    let d10 = Vec2::new(spacing.x, 0.0);
    let d01 = Vec2::new(0.0, spacing.y);
    for y in 0..grid_height + 1 {
        for x in 0..grid_width + 1 {
            let p = Pos2::new(x as f32 * spacing.x, y as f32 * spacing.y);
            painter.line_segment([p, p + d10], stroke);
            painter.line_segment([p, p + d01], stroke);
        }
    }

    // draw masses
    let color = Color32::from_rgba_premultiplied(00, 0x8b, 0x8b, 0x40);
    for position in simulation.state.positions.iter() {
        painter.circle_filled(from_world.transform(position), 3.0, color);
    }
}

fn random_point_in_circle(rng: &mut ThreadRng, radius: f32) -> Pos2 {
    let r = radius * rng.gen::<f32>().sqrt();
    let theta = rng.gen::<f32>() * TAU;
    Pos2::new(r * theta.cos(), r * theta.sin())
}

impl eframe::App for GravityApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {

        let spacing = Vec2::new(32.0, 32.0);
        egui::SidePanel::right("control_panel").show(ctx, |ui| {
            ui.vertical(|ui| {
                ui.checkbox(&mut self.play.on, "play");
                ui.checkbox(&mut self.simulation.barnes_hut, "barnes hut");
                ui.add(egui::Slider::new(&mut self.simulation.theta, 0.0..=100.0).text("theta"));
                
                ui.label("Symplectic integrator");
                if ui.add(egui::RadioButton::new(self.simulation.coefficients == gravity::euler(), "Euler")).clicked() {
                    self.simulation.coefficients = gravity::euler();
                }
                if ui.add(egui::RadioButton::new(self.simulation.coefficients == gravity::leap2(), "Leap 2")).clicked() {
                    self.simulation.coefficients = gravity::leap2();
                }
                if ui.add(egui::RadioButton::new(self.simulation.coefficients == gravity::ruth3(), "Ruth 3")).clicked() {
                    self.simulation.coefficients = gravity::ruth3();
                }
                if ui.add(egui::RadioButton::new(self.simulation.coefficients == gravity::ruth4(), "Ruth 4")).clicked() {
                    self.simulation.coefficients = gravity::ruth4();
                }

            });
            ui.separator();
            ui.vertical(|ui| {
                ui.selectable_value(&mut self.mode, Mode::Add, "Add");
                ui.selectable_value(&mut self.mode, Mode::Remove, "Remove");
            });
            if self.mode == Mode::Add {
                ui.checkbox(&mut self.orbital_velocity, "orbital velocity");
            }
        });
        let id = Id::new("gravity_view");
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.play.on {
                // take a time step
                let dt = ui.input(|i| i.unstable_dt).at_most(1.0 / 30.0);
                self.simulation.step(dt);
    
                // request a new timestep
                ui.ctx().request_repaint();
                //self.time += ;
            };

            let rect = Rect::from_min_size(Pos2::ZERO, ui.available_size());
            
            ui.input(|i| {
                // control zoom
                if let Some(hover_pos) = i.pointer.hover_pos() {
                    self.from_world.zoom(i.zoom_delta(), hover_pos);
                }
                // control circle with mouse
                if i.scroll_delta != Vec2::ZERO {
                    self.radius = (self.radius + 0.1 * i.scroll_delta.y).clamp(0.0, 1024.0);
                }
            });

            draw(ui, &self.simulation, spacing, &self.from_world);

            let response = ui.interact(rect, id, Sense::hover());
            if let Some(hover_pos) = response.hover_pos() {
                let stroke = Stroke { width: 1.0, color: Color32::WHITE};
                ui.painter().circle_stroke(hover_pos, self.radius, stroke);
            }
            
            let response = ui.interact(rect, id, Sense::click_and_drag());
            if response.dragged_by(egui::PointerButton::Primary) {
                if let Some(pointer_pos) = response.interact_pointer_pos() {
                    match self.mode {
                        Mode::Add => {
                            let position = pointer_pos + random_point_in_circle(&mut self.rng, self.radius).to_vec2();
                            let p = self.from_world.inverse(position);
                            let velocity = if self.orbital_velocity {
                                gravity::orbital_velocity(&self.simulation.center_of_mass(), &p)
                            } else {
                                gravity::Vec2::zero()
                            };
                            self.simulation.add(&p, &velocity, 1.0);
                        },
                        Mode::Remove => {
                            let center = self.from_world.inverse(pointer_pos);
                            let r = self.from_world.inversef(self.radius);
                            let r2 = r * r;
                            let mut indices = self.simulation.state.positions
                                .iter()
                                .enumerate()
                                .filter(|(_, &p)| p.sub(&center).norm2() < r2)
                                .map(|(index, _)| index)
                                .collect::<Vec<_>>();
                            indices.reverse();
                            for index in indices {
                                self.simulation.remove(index);
                            }
                        }                            
                    }
                }
            }
            if response.dragged_by(egui::PointerButton::Secondary) {
                self.from_world.pan(response.drag_delta());
            }
        });
    }
}