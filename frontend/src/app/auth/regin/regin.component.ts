import { Component, OnInit } from '@angular/core';
import {AppService} from "../../app.service";
import {Router, Routes} from "@angular/router";

@Component({
  selector: 'app-regin',
  templateUrl: './regin.component.html',
  styleUrls: ['./regin.component.scss']
})
export class ReginComponent implements OnInit {
  public name = '';
  public fname = '';
  public oname = '';
  public email = '';
  public pas = '';
  public req:any;
  public post ='';
  constructor(private service:AppService,private router:Router) { }

  ngOnInit(): void {
  }
  register(){
    this.service.register(this.name,this.fname,this.oname,this.email,this.pas).subscribe(value => {
      console.log(value);
      this.req=value;
      if(this.req.description === "login already in use"){
        alert("Пользователь с таким логином уже существует!!!")
        this.router.navigate(['registration']);
      }
      else{
        this.router.navigate(['auth']);
      }
    })
  }

}
